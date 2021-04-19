/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#include <float.h>
#include <pthread.h>

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/thread.h"

#include "avfilter.h"
#include "filters.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "queue.h"
#include "safe_queue.h"
#include "transpose.h"
#include "video.h"

// Camera projection model
typedef enum CameraModel {

    // Rectilinear projection (`r = f * tan(theta)`)
    CAMERA_MODEL_RECTILINEAR,

    // Equidistant fisheye projection (`r = f * theta`)
    CAMERA_MODEL_EQUIDISTANT_FISHEYE,

    // Number of camera projection models
    NB_CAMERA_MODELS,

} CameraModel;

// Camera properties
typedef struct Camera {

    // Camera projection model
    CameraModel model;

    // Camera focal length in pixels
    double focal_length;

} Camera;

//  Motion stabilization algorithm
typedef enum StabilizationAlgorithm {

    // Do not apply stabilization
    STABILIZATION_ALGORITHM_ORIGINAL,

    // Keep the camera orientation fixed at its orientation in the first frame
    STABILIZATION_ALGORITHM_FIXED,

    // Smooth camera motion with a Savitsky-Golay filter
    STABILIZATION_ALGORITHM_SMOOTH,

    // Number of stabilization algorithms
    NB_STABILIZATION_ALGORITHMS,

} StabilizationAlgorithm;

typedef struct DewobbleOpenCLContext {

    // Generic OpenCL filter context
    OpenCLFilterContext ocf;

    // OpenCL command queue
    cl_command_queue command_queue;

    // Input camera (projection, focal length
    Camera input_camera;

    // Output camera (projection, focal length
    Camera output_camera;

    // Stabilization algorithm applied by the filter
    StabilizationAlgorithm stabilization_algorithm;

    // The number of frames to look ahead and behind for the purpose of stabilizing
    // each frame
    int stabilization_radius;

    // Whether the filter has been initialized
    int initialized;

    // Whether the end of the input link has been reached
    int input_eof;

    // Number of frame jobs currently in progress (read from inlink but
    // not yet sent to outlink)
    int nb_frames_in_progress;
    int nb_frames_consumed;

    // Whether the stabilization thread has been created
    int dewobble_thread_created;

    // Thread where the stabilization transform is done
    pthread_t dewobble_thread;

    // All frames currently in progress
    Queue *frame_queue;

    // frame jobs with the input copied to a buffer, ready for stabilization
    SafeQueue *dewobble_queue;

    // frames jobs with a stabilized buffer, ready to be copied to the output frame
    SafeQueue *output_queue;

} DewobbleOpenCLContext;

// Stages for the filtering of a frame
typedef enum FrameJobStage {

    // Input frame OpenCL images to be copied to an OpenCL buffer
    FRAME_JOB_STAGE_READING_INPUT,

    // Input frame OpenCL buffer to have dewobbling applied
    FRAME_JOB_STAGE_DEWOBBLING,

    //  Output OpenCL buffer to be copied to output frame OpenCL images
    FRAME_JOB_STAGE_WRITING_OUTPUT,

    // Work complete, output frame ready to be sent to output link
    FRAME_JOB_STAGE_DONE

} FrameJobStage;

// The state of work associated with a particular frame
typedef struct FrameJob {
    int num;

    // The AVFilterContext for the filter
    AVFilterContext *avctx;

    // The current stage of processing
    FrameJobStage stage;

    // The error associated with a particular frame (or EOF after the last frame)
    int err;

    // Frame presentation timestamp
    int64_t pts;

    // Input frame
    AVFrame *input_frame;

    // Output frame
    AVFrame *output_frame;

    // Input frame as OpenCL buffer
    cl_mem input_buffer;

    // Output frame as OpenCL buffer
    cl_mem output_buffer;

    // Mutex protecting access to `stage` and `copy_events_remaining`
    AVMutex mutex;

    // For stages involving copying between OpenCL images and a buffer, the number
    // of OpenCL events remaining before the next stage can be started
    int copy_events_remaining;

} FrameJob;

typedef enum DewobbleMessageType {
    DEWOBBLE_MESSAGE_TYPE_JOB,
    DEWOBBLE_MESSAGE_TYPE_EOF
} DewobbleMessageType;

typedef struct DewobbleMessage {
    DewobbleMessageType type;
    FrameJob *job;
} DewobbleMessage;

#define EXTRA_IN_PROGRESS_FRAMES 2

static FrameJobStage queued_frame_get_stage(FrameJob *job) {
    FrameJobStage stage;
    ff_mutex_lock(&job->mutex);
    stage = job->stage;
    ff_mutex_unlock(&job->mutex);
    return stage;
};

static void frame_job_set_stage(FrameJob *job, FrameJobStage stage) {
    ff_mutex_lock(&job->mutex);
    job->stage = stage;
    ff_mutex_unlock(&job->mutex);
};

static DewobbleMessage *dewobble_message_create(DewobbleMessageType type, FrameJob *job) {
    DewobbleMessage *result = av_mallocz(sizeof(DewobbleMessage));
    if (result == NULL) {
        return result;
    }
    result->type = type;
    result->job = job;
    return result;
}

static void *dewobble_thread(void *arg) {
    AVFilterContext *avctx = arg;
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *message = NULL;
    FrameJob *job;
    int is_eof = 0;

    av_log(avctx, AV_LOG_VERBOSE, "Started worker thread\n");
    while (!is_eof) {
        message = ff_safe_queue_pop_front_blocking(ctx->dewobble_queue);

        if (message->type == DEWOBBLE_MESSAGE_TYPE_JOB) {
            // TODO: apply dewobbling
            job = message->job;
            job->output_buffer = job->input_buffer;
            job->input_buffer = NULL;

            frame_job_set_stage(job, FRAME_JOB_STAGE_WRITING_OUTPUT);
            ff_safe_queue_push_back(ctx->output_queue, job);
            ff_filter_set_ready(avctx, 1);
            av_log(avctx, AV_LOG_VERBOSE, "Worker thread: pushed frame %d\n", job->num);
        } else if (message->type == DEWOBBLE_MESSAGE_TYPE_EOF) {
            is_eof = 1;
        }
        free(message);
    }

    av_log(avctx, AV_LOG_VERBOSE, "Worker thread: reached EOF, exiting.\n");
    return NULL;
}

static int dewobble_opencl_init(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    ctx->frame_queue = ff_queue_create();
    if (ctx->frame_queue == NULL) {
        return AVERROR(ENOMEM);
    }
    ctx->dewobble_queue = ff_safe_queue_create();
    if (ctx->dewobble_queue == NULL) {
        return AVERROR(ENOMEM);
    }
    ctx->output_queue = ff_safe_queue_create();
    if (ctx->output_queue == NULL) {
        return AVERROR(ENOMEM);
    }
    return ff_opencl_filter_init(avctx);
}

static void dewobble_opencl_uninit(AVFilterContext *avctx) {
    cl_int cle;
    DewobbleOpenCLContext *ctx = avctx->priv;
    if (ctx->dewobble_thread_created)
    {
        pthread_join(ctx->dewobble_thread, NULL);
        ctx->dewobble_thread_created = 0;
    }
    ff_queue_destroy(ctx->frame_queue);
    ff_safe_queue_destroy(ctx->dewobble_queue);
    ff_safe_queue_destroy(ctx->output_queue);
    if (ctx->command_queue) {
        cle = clReleaseCommandQueue(ctx->command_queue);
        if (cle != CL_SUCCESS) {
            av_log(avctx, AV_LOG_ERROR, "Failed to release command queue: %d.\n", cle);
        }
    }
    ff_opencl_filter_uninit(avctx);
}

static int dewobble_opencl_frames_init(AVFilterContext *avctx)
{
    DewobbleOpenCLContext *ctx = avctx->priv;
    cl_int cle;
    int err;

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    if (cle) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create OpenCL command queue %d.\n", cle);
        return AVERROR(EIO);
    }

    err = pthread_create(&ctx->dewobble_thread, NULL, dewobble_thread, avctx);
    if (err) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create dewobble thread %d.\n", err);
        return AVERROR(ENOMEM);
    }
    ctx->dewobble_thread_created = 1;

    ctx->initialized = 1;
    return 0;
}

static int dewobble_opencl_config_input(AVFilterLink *inlink) {
    AVFilterContext   *avctx = inlink->dst;
    DewobbleOpenCLContext *ctx = avctx->priv;
    int ret;

    ret = ff_opencl_filter_config_input(inlink);
    if (ret < 0) {
        return ret;
    }

    if (ctx->ocf.output_format != AV_PIX_FMT_NV12) {
        av_log(avctx, AV_LOG_ERROR, "Only NV12 input is supported!\n");
        return AVERROR(ENOSYS);
    }

    // TODO: change output width/height
    return 0;
}

static FrameJob *frame_job_create(
    AVFilterContext *avctx,
    int err,
    int64_t pts,
    AVFrame *input_frame
) {
    FrameJob *result = av_mallocz(sizeof(FrameJob));
    if (result == NULL) {
        return NULL;
    }
    result->avctx = avctx;
    result->err = err;
    result->pts = pts;
    result->input_frame = input_frame;
    ff_mutex_init(&result->mutex, NULL);
    result->stage = FRAME_JOB_STAGE_READING_INPUT;
    return result;
}

static void frame_job_free(FrameJob **frame) {
    if (*frame != NULL) {
        av_frame_free(&(*frame)->input_frame);
        av_frame_free(&(*frame)->output_frame);
        clReleaseMemObject((*frame)->input_buffer);
        clReleaseMemObject((*frame)->output_buffer);
        ff_mutex_destroy(&(*frame)->mutex);
    }
    av_freep(frame);
}

static int frame_job_decrement_events_remaining(FrameJob *job) {
    int remaining;
    ff_mutex_lock(&job->mutex);
    remaining = --job->copy_events_remaining;
    ff_mutex_unlock(&job->mutex);
    return remaining;
}

static void on_buffer_create_progress(cl_event event, cl_int ret, void *data) {
    FrameJob *job = (FrameJob *) data;
    DewobbleMessage *message;
    DewobbleOpenCLContext *ctx = (DewobbleOpenCLContext *) job->avctx->priv;
    int events_remaining = frame_job_decrement_events_remaining(job);

    if (events_remaining == 0) {
        av_log(job->avctx, AV_LOG_VERBOSE, "Finished creating buffer %d\n", job->num);
        message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_JOB, job);
        ret = ff_safe_queue_push_back(ctx->dewobble_queue, message);
        if (ret < 0)
        {
            // TODO errors?
        }
    }
}

static void on_frame_create_progress(cl_event event, cl_int ret, void *data) {
    FrameJob *job = (FrameJob *) data;
    int events_remaining = frame_job_decrement_events_remaining(job);

    if (events_remaining == 0) {
        av_log(job->avctx, AV_LOG_VERBOSE, "Finished creating output frame %d\n", job->num);
        frame_job_set_stage(job, FRAME_JOB_STAGE_DONE);
        ff_filter_set_ready(job->avctx, 1);
    }
}

static cl_int start_copy_frame_to_buffer(
    AVFilterContext *avctx,
    cl_context context,
    cl_command_queue command_queue,
    FrameJob *job
) {
    AVFrame *frame = job->input_frame;
    cl_mem luma = (cl_mem) frame->data[0];
    cl_mem chroma = (cl_mem) frame->data[1];
    cl_int ret = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t luma_region[3] = { frame->width, frame->height, 1 };
    size_t chroma_region[3] = { frame->width / 2, frame->height / 2, 1 };
    cl_event luma_finished;
    cl_event chroma_finished;

    job->input_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        frame->width * frame->height * 3 / 2,
        NULL,
        &ret
    );
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create buffer: %d\n", ret);
        return ret;
    }

    ret = clEnqueueCopyImageToBuffer(
        command_queue,
        luma,
        job->input_buffer,
        src_origin,
        luma_region,
        0,
        0,
        NULL,
        &luma_finished
    );
    ret |= clEnqueueCopyImageToBuffer(
        command_queue,
        chroma,
        job->input_buffer,
        src_origin,
        chroma_region,
        frame->width * frame->height * 1,
        0,
        NULL,
        &chroma_finished
    );
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to copy images to buffer: %d\n", ret);
        clReleaseMemObject(job->input_buffer);
        job->input_buffer = NULL;
        return ret;
    }

    job->copy_events_remaining = 2;
    ret = clSetEventCallback(
        luma_finished,
        CL_COMPLETE,
        on_buffer_create_progress,
        job
    );
    ret |= clSetEventCallback(
        chroma_finished,
        CL_COMPLETE,
        on_buffer_create_progress,
        job
    );

    return ret;
}

static int start_copy_buffer_to_frame(
    AVFilterContext *avctx,
    AVFilterLink *outlink,
    FrameJob *job
) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    AVFrame *frame = job->output_frame;
    cl_mem luma = (cl_mem) frame->data[0];
    cl_mem chroma = (cl_mem) frame->data[1];
    cl_int ret = 0;
    size_t dst_origin[3] = { 0, 0, 0 };
    size_t luma_region[3] = { frame->width, frame->height, 1 };
    size_t chroma_region[3] = { frame->width / 2, frame->height / 2, 1 };
    cl_event luma_finished;
    cl_event chroma_finished;

    ret = clEnqueueCopyBufferToImage(
        ctx->command_queue,
        job->output_buffer,
        luma,
        0,
        dst_origin,
        luma_region,
        0,
        NULL,
        &luma_finished
    );
    ret |= clEnqueueCopyBufferToImage(
        ctx->command_queue,
        job->output_buffer,
        chroma,
        frame->width * frame->height * 1,
        dst_origin,
        chroma_region,
        0,
        NULL,
        &chroma_finished
    );
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to copy buffer to images: %d\n", ret);
        return ret;
    }

    job->copy_events_remaining = 2;
    ret = clSetEventCallback(
        luma_finished,
        CL_COMPLETE,
        on_frame_create_progress,
        job
    );
    ret |= clSetEventCallback(
        chroma_finished,
        CL_COMPLETE,
        on_frame_create_progress,
        job
    );

    return ret;
}

static int consume_input_frame(AVFilterContext *avctx, AVFrame *input_frame) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    FrameJob *job = NULL;
    int ret = 0;

    if (!input_frame->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!ctx->initialized) {
        av_log(avctx, AV_LOG_VERBOSE, "Initializing\n");
        ret = dewobble_opencl_frames_init(avctx);
        if (ret < 0) {
            return ret;
        }
    }

    job = frame_job_create(
        avctx,
        0,
        input_frame->pts,
        input_frame
    );
    job->num = ctx->nb_frames_consumed;
    if (job == NULL) {
        return AVERROR(ENOMEM);
    }
    ff_queue_push_back(ctx->frame_queue, job);

    ret = start_copy_frame_to_buffer(
        avctx,
        ctx->ocf.hwctx->context,
        ctx->command_queue,
        job
    );
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to map OpenCL frame to OpenCL buffer: %d\n", ret);
        return AVERROR(EINVAL);
    }
    return 0;
}

static int input_frame_wanted(DewobbleOpenCLContext *ctx) {
    return !ctx->input_eof && ctx->nb_frames_in_progress < ctx->stabilization_radius
        + 1 + EXTRA_IN_PROGRESS_FRAMES;
}

static int try_copy_output_buffer_to_frame(
    AVFilterContext *avctx,
    DewobbleOpenCLContext *ctx,
    AVFilterLink *outlink
) {
    int ret = 0;
    FrameJob * job = (FrameJob *) ff_safe_queue_pop_front(ctx->output_queue);

    if (job != NULL) {
        job->output_frame = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        av_frame_copy_props(job->output_frame, job->input_frame);
        ret = start_copy_buffer_to_frame(avctx, outlink, job);
    }
    return ret;
}

static int try_send_output_frame(
    AVFilterContext *avctx,
    DewobbleOpenCLContext *ctx,
    AVFilterLink *inlink,
    AVFilterLink *outlink
) {
    int ret = 0;
    FrameJob *job = (FrameJob *) ff_queue_peek_front(ctx->frame_queue);
    DewobbleMessage *message;
    AVFrame *output_frame;

    if (job != NULL && queued_frame_get_stage(job) == FRAME_JOB_STAGE_DONE) {
        ff_queue_pop_front(ctx->frame_queue);
        if (job->err) {
            ret = job->err;
        } else {
            av_log(
                avctx,
                AV_LOG_VERBOSE,
                "Sending output frame %d (%d in progress)\n",
                job->num,
                ctx->nb_frames_in_progress
            );
            ctx->nb_frames_in_progress -= 1;
            output_frame = av_frame_clone(job->output_frame);
            ret = ff_filter_frame(outlink, output_frame);
            if (ret) {
                return ret;
            }

            if (input_frame_wanted(ctx)) {
                ff_inlink_request_frame(inlink);
            }

            if (ctx->input_eof && ctx->nb_frames_in_progress == 0) {
                av_log(avctx, AV_LOG_VERBOSE, "Output reached EOF\n");
                message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_EOF, NULL);
                ff_safe_queue_push_back(ctx->dewobble_queue, message);
                ff_outlink_set_status(outlink, AVERROR_EOF, job->pts);
            }
        }
        frame_job_free(&job);
    }
    return ret;
}

static int try_consume_input_frame(
    AVFilterContext *avctx,
    DewobbleOpenCLContext *ctx,
    AVFilterLink *inlink
) {
    int ret = 0;
    AVFrame *input_frame;
    // If necessary, attempt to consume a frame from the input
    if (input_frame_wanted(ctx)) {
        ret = ff_inlink_consume_frame(inlink, &input_frame);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to read input frame\n");
            return ret;
        } else if (ret > 0) {
            av_log(
                avctx,
                AV_LOG_VERBOSE,
                "Consuming input frame %d (%d in progress)\n",
                ctx->nb_frames_consumed,
                ctx->nb_frames_in_progress
            );
            ret = consume_input_frame(avctx, input_frame);
            ctx->nb_frames_in_progress += 1;
            ctx->nb_frames_consumed += 1;
            if (ret) {
                av_log(avctx, AV_LOG_ERROR, "Failed to consume input frame\n");
                av_frame_free(&input_frame);
                return ret;
            }
        }

        // If we still need more input frames, request them now
        if (input_frame_wanted(ctx)) {
            ff_inlink_request_frame(inlink);
        }
    }
    return ret;
}

static int check_for_input_eof(
    AVFilterContext *avctx,
    DewobbleOpenCLContext *ctx,
    AVFilterLink *inlink,
    AVFilterLink *outlink
) {
    int64_t pts;
    int ret = 0;
    int status;
    DewobbleMessage *message;

    // Check for end of input
    if (!ctx->input_eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            av_log(avctx, AV_LOG_VERBOSE, "Reached input EOF\n");
            ctx->input_eof = 1;
            if (ctx->nb_frames_in_progress == 0) {
                message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_EOF, NULL);
                ff_safe_queue_push_back(ctx->dewobble_queue, message);
                ff_outlink_set_status(outlink, AVERROR_EOF, pts);
            }
        } else if (status) {
            av_log(avctx, AV_LOG_ERROR, "INPUT STATUS: %d\n", status);
        }
    }
    return ret;
}

static int activate(AVFilterContext *avctx)
{
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *outlink = avctx->outputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    int ret = 0;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    ret = try_consume_input_frame(avctx, ctx, inlink);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "CONSUME INPUT FAILED: %d\n", ret);
        return ret;
    }

    ret = check_for_input_eof(avctx, ctx, inlink, outlink);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "CHECK INPUT STATUS FAILED: %d\n", ret);
        return ret;
    }

    ret = try_copy_output_buffer_to_frame(avctx, ctx, outlink);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "COPY OUTPUT BUFFER FAILED: %d\n", ret);
        return ret;
    }

    ret = try_send_output_frame(avctx, ctx, inlink, outlink);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "SEND OUTPUT FRAME FAILED: %d\n", ret);
        return ret;
    }

    return FFERROR_NOT_READY;
}


#define OFFSET(x) offsetof(DewobbleOpenCLContext, x)
#define OFFSET_CAMERA(x) offsetof(Camera, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption dewobble_opencl_options[] = {
    // Input camera options
    {
        "input_model",
        "input camera projection model",
        OFFSET(input_camera) + OFFSET_CAMERA(model),
        AV_OPT_TYPE_INT,
        { .i64 = NB_CAMERA_MODELS },
        0,
        NB_CAMERA_MODELS - 1,
        FLAGS,
        "model",
    },
    {
        "input_focal",
        "input camera focal length in pixels",
        OFFSET(input_camera) + OFFSET_CAMERA(focal_length),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = 0 },
        0,
        DBL_MAX,
        .flags=FLAGS,
        .unit = "in_f",
    },

    // Output camera options
    {
        "output_model",
        "output camera projection model",
        OFFSET(output_camera) + OFFSET_CAMERA(model),
        AV_OPT_TYPE_INT,
        { .i64 = NB_CAMERA_MODELS },
        0,
        NB_CAMERA_MODELS - 1,
        FLAGS,
        "model",
    },
    {
        "output_focal",
        "output camera focal length in pixels",
        OFFSET(output_camera) + OFFSET_CAMERA(focal_length),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = 0 },
        0,
        DBL_MAX,
        .flags=FLAGS,
        .unit = "out_f",
    },

    // Stabilization options
    {
        "stab",
        "camera orientation stabilization algorithm",
        OFFSET(stabilization_algorithm),
        AV_OPT_TYPE_INT,
        { .i64 = STABILIZATION_ALGORITHM_SMOOTH },
        0,
        NB_STABILIZATION_ALGORITHMS - 1,
        FLAGS,
        "stab",
    },
    {
        "stab_radius",
        "for Savitzky-Golay smoothing: the number of frames to look ahead and behind",
        OFFSET(stabilization_radius),
        AV_OPT_TYPE_INT,
        { .i64 = 15 },
        1,
        INT_MAX,
        FLAGS,
        "radius",
    },

    // Camera models
    {
        "rectilinear",
        "rectilinear projection",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = CAMERA_MODEL_RECTILINEAR },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "model"
    },
    {
        "fisheye",
        "equidistant fisheye projection",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = CAMERA_MODEL_EQUIDISTANT_FISHEYE },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "model"
    },

    // Stabilization algorithms
    {
        "fixed",
        "fix the camera orientation after the first frame",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = STABILIZATION_ALGORITHM_FIXED },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "stab"
    },
    {
        "original",
        "maintain the camera orientation from the input (do not apply stabilization)",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = STABILIZATION_ALGORITHM_ORIGINAL },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "stab"
    },
    {
        "smooth",
        "smooth the camera orientation using a Savitzky-Golay filter",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = STABILIZATION_ALGORITHM_FIXED },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "stab"
    },

    { NULL }
};

AVFILTER_DEFINE_CLASS(dewobble_opencl);

static const AVFilterPad dewobble_opencl_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &dewobble_opencl_config_input,
    },
    { NULL }
};

static const AVFilterPad dewobble_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &ff_opencl_filter_config_output,
    },
    { NULL }
};

AVFilter ff_vf_dewobble_opencl = {
    .name           = "dewobble_opencl",
    .description    = NULL_IF_CONFIG_SMALL(
        "apply motion stabilization with awareness of camera projection and/or change camera projection"
    ),
    .priv_size      = sizeof(DewobbleOpenCLContext),
    .priv_class     = &dewobble_opencl_class,
    .init           = &dewobble_opencl_init,
    .uninit         = &dewobble_opencl_uninit,
    .query_formats  = &ff_opencl_filter_query_formats,
    .inputs         = dewobble_opencl_inputs,
    .outputs        = dewobble_opencl_outputs,
    .activate       = activate,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
