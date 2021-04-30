/*
 * Copyright (c) 2021 Daniel Playfair Cal <daniel.playfair.cal@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <float.h>
#include <pthread.h>
#include <signal.h>
#include <dewobble/dewobble.h>

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

// Camera properties
typedef struct Camera {

    // Camera projection model
    int model;

    // Camera focal length in pixels
    double focal_length;

    // Width in pixels
    int width;

    // Height in pixels
    int height;

    // Horizonal coordinate of focal point in pixels
    double focal_point_x;

    // Vertical coordinate of focal point in pixels
    double focal_point_y;
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
    int stabilization_algorithm;

    // The number of frames to look ahead and behind for the purpose of stabilizing
    // each frame
    int stabilization_radius;

    // The number of frames to look ahead for the purpose of interpolating
    // frame rotation for frames where detection fails
    int stabilization_horizon;

    // The algorithm to interpolate the value between source image pixels
    int interpolation_algorithm;

    // Whether the filter has been initialized
    int initialized;

    // Whether the end of the input link has been reached
    int input_eof;

    // Number of frame jobs currently in progress (read from inlink but
    // not yet sent to outlink)
    int nb_frames_in_progress;

    // Number of frames consumed so far
    long nb_frames_consumed;

    // Whether the stabilization thread has been created
    int dewobble_thread_created;

    // Whether the stabilization thread has been cancelled
    int dewobble_thread_ending;

    // Thread where the stabilization transform is done
    pthread_t dewobble_thread;

    // frame jobs with the input copied to a buffer, ready for stabilization
    SafeQueue *dewobble_queue;

    // frames jobs with a stabilized buffer, ready to be copied to the output frame
    SafeQueue *output_queue;

} DewobbleOpenCLContext;

// The state of work associated with a particular frame
typedef struct FrameJob {
    // Serial number for this frame
    long num;

    // The AVFilterContext for the filter
    AVFilterContext *avctx;

    // Frame presentation timestamp
    int64_t pts;

    // Input frame
    AVFrame *input_frame;

    // Input frame as OpenCL buffer
    cl_mem input_buffer;

    // Output frame as OpenCL buffer
    cl_mem output_buffer;

} FrameJob;

typedef enum DewobbleMessageType {
    DEWOBBLE_MESSAGE_TYPE_JOB,
    DEWOBBLE_MESSAGE_TYPE_EOF,
    DEWOBBLE_MESSAGE_TYPE_ERROR
} DewobbleMessageType;

typedef struct DewobbleMessage {
    DewobbleMessageType type;
    FrameJob *job;
} DewobbleMessage;

#define EXTRA_IN_PROGRESS_FRAMES 2

static DewobbleMessage *dewobble_message_create(DewobbleMessageType type, FrameJob *job) {
    DewobbleMessage *result = av_mallocz(sizeof(DewobbleMessage));
    if (result == NULL) {
        return result;
    }
    result->type = type;
    result->job = job;
    return result;
}

static void dewobble_message_free(DewobbleMessage **message) {
    if (*message != NULL) {
        free(*message);
        *message = NULL;
    }
}

static FrameJob *frame_job_create(
    AVFilterContext *avctx,
    int64_t pts,
    AVFrame *input_frame
) {
    FrameJob *result = av_mallocz(sizeof(FrameJob));
    if (result == NULL) {
        return NULL;
    }
    result->avctx = avctx;
    result->pts = pts;
    result->input_frame = input_frame;
    return result;
}

static void frame_job_free(AVFilterContext *avctx, FrameJob **frame) {
    cl_int cle;
    if (*frame != NULL) {
        av_frame_free(&(*frame)->input_frame);
        CL_RELEASE_MEMORY((*frame)->input_buffer);
        CL_RELEASE_MEMORY((*frame)->output_buffer);
    }
    av_freep(frame);
}

static void flush_queues(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *message = NULL;

    // Flush input queue
    while (message = ff_safe_queue_pop_front(ctx->dewobble_queue)) {
        if (message->type == DEWOBBLE_MESSAGE_TYPE_JOB) {
            frame_job_free(avctx, &message->job);
        }
        dewobble_message_free(&message);
    }

    // Flush output queue
    while (message = ff_safe_queue_pop_front(ctx->output_queue)) {
        if (message->type == DEWOBBLE_MESSAGE_TYPE_JOB) {
            frame_job_free(avctx, &message->job);
        }
        dewobble_message_free(&message);
    }
}

static int pull_ready_frame(AVFilterContext *avctx, DewobbleFilter filter) {
    int err;
    DewobbleOpenCLContext *ctx = avctx->priv;
    cl_mem output_buffer = NULL;
    FrameJob *job = NULL;
    DewobbleMessage *output_message = NULL;

    err = dewobble_filter_pull_frame(filter, &output_buffer, (void **) &job);
    if (err) {
        av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to pull frame from dewobbler\n");
        goto fail;
    }
    job->output_buffer = output_buffer;

    output_message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_JOB, job);
    if (output_message == NULL) {
        goto fail;
    }

    av_log(avctx, AV_LOG_VERBOSE, "Worker thread: sent frame %ld\n", job->num);
    err = ff_safe_queue_push_back(ctx->output_queue, output_message);
    if (err == -1) {
        goto fail;
    }
    return 0;

fail:
    clReleaseMemObject(output_buffer);
    dewobble_message_free(&output_message);
    return err;
}

static int pull_ready_frames(
    AVFilterContext *avctx,
    DewobbleFilter filter
) {
    int err = 0;

    while (dewobble_filter_frame_ready(filter)) {
        err = pull_ready_frame(avctx, filter);
        if (err) {
            return err;
        }
    }
    ff_filter_set_ready(avctx, 1);

    return err;
}

static void *dewobble_thread(void *arg) {
    AVFilterContext *avctx = arg;
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *dewobble_message = NULL,  *output_message = NULL;
    FrameJob *job;
    DewobbleFilter filter = NULL;
    DewobbleCamera input_camera = NULL, output_camera = NULL;
    DewobbleStabilizer stabilizer = NULL;
    int input_eof = 0;
    int err = 0;

    dewobble_init_opencl_context(ctx->ocf.hwctx->context, ctx->ocf.hwctx->device_id);

    input_camera = dewobble_camera_create(
        ctx->input_camera.model,
        ctx->input_camera.focal_length,
        ctx->input_camera.width,
        ctx->input_camera.height,
        ctx->input_camera.focal_point_x,
        ctx->input_camera.focal_point_y
    );
    output_camera = dewobble_camera_create(
        ctx->output_camera.model,
        ctx->output_camera.focal_length,
        ctx->output_camera.width,
        ctx->output_camera.height,
        ctx->output_camera.focal_point_x,
        ctx->output_camera.focal_point_y
    );
    switch(ctx->stabilization_algorithm) {
        case STABILIZATION_ALGORITHM_ORIGINAL:
            stabilizer = dewobble_stabilizer_create_none();
            break;
         case STABILIZATION_ALGORITHM_FIXED:
            stabilizer = dewobble_stabilizer_create_fixed(
                input_camera,
                ctx->stabilization_horizon
            );
            break;
        case STABILIZATION_ALGORITHM_SMOOTH:
            stabilizer = dewobble_stabilizer_create_savitzky_golay(
                input_camera,
                ctx->stabilization_radius,
                ctx->stabilization_horizon
            );
            break;
    }
    filter = dewobble_filter_create(
        input_camera,
        output_camera,
        stabilizer,
        ctx->interpolation_algorithm
    );
    if (filter == NULL) {
        av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to create libdewobble filter\n");
        err = 1;
    }

    while (!err && !input_eof) {
        dewobble_message = ff_safe_queue_pop_front_blocking(ctx->dewobble_queue);

        switch (dewobble_message->type) {
            case DEWOBBLE_MESSAGE_TYPE_JOB:
                job = dewobble_message->job;

                av_log(avctx, AV_LOG_VERBOSE, "Worker thread: transforming frame %ld\n", job->num);
                err = dewobble_filter_push_frame(filter, job->input_buffer, job);
                if (err) {
                    frame_job_free(avctx, &job);
                    av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to push %ld\n", job->num);
                    break;
                }
                job = NULL;
                err = pull_ready_frames(avctx, filter);
                if (err) {
                    av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to pull frames\n");
                }
                break;
            case DEWOBBLE_MESSAGE_TYPE_EOF:
                av_log(avctx, AV_LOG_VERBOSE, "Worker thread: reached end of input\n");
                input_eof = 1;
                err = dewobble_filter_end_input(filter);
                if (err) {
                    av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to end input\n");
                    break;
                }
                err = pull_ready_frames(avctx, filter);
                if (err) {
                    av_log(avctx, AV_LOG_ERROR, "Worker thread: failed to pull frames\n");
                }
                break;
            case DEWOBBLE_MESSAGE_TYPE_ERROR:
                err = 1;
                break;
        }
        dewobble_message_free(&dewobble_message);
    }

    if (err) {
        output_message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_ERROR, NULL);
        if (output_message != NULL) {
            ff_safe_queue_push_back(ctx->output_queue, output_message);
            ff_filter_set_ready(avctx, 1);
        }
    }

    dewobble_filter_destroy(filter);
    dewobble_camera_destroy(input_camera);
    dewobble_camera_destroy(output_camera);
    return NULL;
}

static void send_eof_to_dewobble_thread(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *message = NULL;
    int err;

    if (ctx->dewobble_thread_ending == 1) {
        return;
    }
    ctx->dewobble_thread_ending = 1;

    message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_EOF, NULL);
    if (message == NULL) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    err = ff_safe_queue_push_back(ctx->dewobble_queue, message);
    if (err == -1) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    return;

fail:
    av_log(avctx, AV_LOG_ERROR, "Failed to send EOF to dewobble thread: %d\n", err);
    dewobble_message_free(&message);
}

static void stop_dewobble_thread_on_error(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *message = NULL;
    int err;

    if (ctx->dewobble_thread_ending == 1) {
        return;
    }
    ctx->dewobble_thread_ending = 1;

    message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_ERROR, NULL);
    if (message == NULL) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    err = ff_safe_queue_push_front(ctx->dewobble_queue, message);
    if (err == -1) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    return;

fail:
    av_log(avctx, AV_LOG_ERROR, "Failed to send stop message to dewobble thread: %d\n", err);
    dewobble_message_free(&message);
}

static int dewobble_opencl_init(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    if (ctx->input_camera.model == DEWOBBLE_NB_PROJECTIONS
        || ctx->output_camera.model == DEWOBBLE_NB_PROJECTIONS
    ) {
        av_log(avctx, AV_LOG_ERROR, "both in_p and out_p must be set\n");
        return AVERROR(EINVAL);
    }
    if (ctx->input_camera.focal_length == 0 || ctx->output_camera.focal_length == 0) {
        av_log(avctx, AV_LOG_ERROR, "both in_fl and out_fl must be set\n");
        return AVERROR(EINVAL);
    }
    if (ctx->stabilization_algorithm == STABILIZATION_ALGORITHM_ORIGINAL) {
        ctx->stabilization_horizon = 0;
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
    av_log(avctx, AV_LOG_VERBOSE, "Uninit\n");
    if (ctx->dewobble_thread_created)
    {
        stop_dewobble_thread_on_error(avctx);
        pthread_join(ctx->dewobble_thread, NULL);
        ctx->dewobble_thread_created = 0;
        flush_queues(avctx);
    }
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

    ctx->input_camera.width = inlink->w;
    ctx->input_camera.height = inlink->h;

    // Output camera defaults to the same resolution as the input
    if (ctx->output_camera.width == 0) {
        ctx->output_camera.width = ctx->input_camera.width;
    }
    if (ctx->output_camera.height == 0) {
        ctx->output_camera.height = ctx->input_camera.height;
    }
    ctx->ocf.output_width = ctx->output_camera.width;
    ctx->ocf.output_height = ctx->output_camera.height;

    // Focal points default to the image center
    if (ctx->input_camera.focal_point_x == DBL_MAX) {
        ctx->input_camera.focal_point_x = (ctx->input_camera.width - 1) / 2.0;
    }
    if (ctx->input_camera.focal_point_y == DBL_MAX) {
        ctx->input_camera.focal_point_y = (ctx->input_camera.height - 1) / 2.0;
    }
    if (ctx->output_camera.focal_point_x == DBL_MAX) {
        ctx->output_camera.focal_point_x = (ctx->output_camera.width - 1) / 2.0;
    }
    if (ctx->output_camera.focal_point_y == DBL_MAX) {
        ctx->output_camera.focal_point_y = (ctx->output_camera.height - 1) / 2.0;
    }

    return 0;
}

static cl_int copy_frame_to_buffer(
    AVFilterContext *avctx,
    cl_context context,
    cl_command_queue command_queue,
    FrameJob *job
) {
    int err;
    AVFrame *frame = job->input_frame;
    cl_mem luma = (cl_mem) frame->data[0];
    cl_mem chroma = (cl_mem) frame->data[1];
    cl_int cle = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t luma_region[3] = { frame->width, frame->height, 1 };
    size_t chroma_region[3] = { frame->width / 2, frame->height / 2, 1 };
    cl_event copy_finished[2];

    job->input_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        frame->width * frame->height * 3 / 2,
        NULL,
        &cle
    );
    CL_FAIL_ON_ERROR(AVERROR(ENOMEM), "Failed to create buffer: %d\n", cle);

    cle = clEnqueueCopyImageToBuffer(
        command_queue,
        luma,
        job->input_buffer,
        src_origin,
        luma_region,
        0,
        0,
        NULL,
        &copy_finished[0]
    );
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to enqueue copy luma image to buffer: %d\n", cle);

    cle = clEnqueueCopyImageToBuffer(
        command_queue,
        chroma,
        job->input_buffer,
        src_origin,
        chroma_region,
        frame->width * frame->height * 1,
        0,
        NULL,
        &copy_finished[1]
    );
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to enqueue copy chroma image to buffer: %d\n", cle);

    cle = clWaitForEvents(2, copy_finished);
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to copy images to buffer: %d\n", cle);

    return 0;

fail:
    CL_RELEASE_MEMORY(job->input_buffer);
    job->input_buffer = NULL;
    return err;
}

static int copy_buffer_to_frame(
    AVFilterContext *avctx,
    cl_mem buffer,
    AVFrame *output_frame
) {
    int err;
    DewobbleOpenCLContext *ctx = avctx->priv;
    cl_mem luma = (cl_mem) output_frame->data[0];
    cl_mem chroma = (cl_mem) output_frame->data[1];
    cl_int cle = 0;
    size_t dst_origin[3] = { 0, 0, 0 };
    size_t luma_region[3] = { output_frame->width, output_frame->height, 1 };
    size_t chroma_region[3] = { output_frame->width / 2, output_frame->height / 2, 1 };
    cl_event copy_finished[2];

    cle = clEnqueueCopyBufferToImage(
        ctx->command_queue,
        buffer,
        luma,
        0,
        dst_origin,
        luma_region,
        0,
        NULL,
        &copy_finished[0]
    );
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to enqueue copy buffer to luma image: %d\n", cle);
    cle = clEnqueueCopyBufferToImage(
        ctx->command_queue,
        buffer,
        chroma,
        output_frame->width * output_frame->height * 1,
        dst_origin,
        chroma_region,
        0,
        NULL,
        &copy_finished[1]
    );
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to enqueue copy buffer to luma image: %d\n", cle);

    cle = clWaitForEvents(2, copy_finished);
    CL_FAIL_ON_ERROR(AVERROR(EINVAL), "Failed to copy buffer to images: %d\n", cle);

    return 0;

fail:
    return err;
}

static int consume_input_frame(AVFilterContext *avctx, AVFrame *input_frame) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    FrameJob *job = NULL;
    DewobbleMessage *message = NULL;
    int err = 0;
    int i;

    if (!input_frame->hw_frames_ctx) {
        return AVERROR(EINVAL);
    }

    if (input_frame->crop_top || input_frame->crop_bottom ||
        input_frame->crop_left || input_frame->crop_right
    ) {
        av_log(
            avctx,
            AV_LOG_WARNING,
            "Cropping information discarded from input (code not written yet)\n"
        );
    }

    if (!ctx->initialized) {
        av_log(avctx, AV_LOG_VERBOSE, "Initializing\n");
        err = dewobble_opencl_frames_init(avctx);
        if (err < 0) {
            return err;
        }
    }

    job = frame_job_create(
        avctx,
        input_frame->pts,
        input_frame
    );
    job->num = ctx->nb_frames_consumed;
    if (job == NULL) {
        return AVERROR(ENOMEM);
    }

    err = copy_frame_to_buffer(
        avctx,
        ctx->ocf.hwctx->context,
        ctx->command_queue,
        job
    );
    if (err) {
        goto fail;
    }

    // Free original input frame buffers
    for (i = 0; input_frame->buf[i] != NULL; i++) {
        av_buffer_unref(&input_frame->buf[i]);
    }

    message = dewobble_message_create(DEWOBBLE_MESSAGE_TYPE_JOB, job);
    err = ff_safe_queue_push_back(ctx->dewobble_queue, message);
    if (err == -1) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->nb_frames_in_progress += 1;
    ctx->nb_frames_consumed += 1;

    return 0;

fail:
    frame_job_free(avctx, &job);
    dewobble_message_free(&message);
    return err;
}

static int input_frame_wanted(DewobbleOpenCLContext *ctx) {
    int nb_buffered_frames = ctx->stabilization_algorithm == STABILIZATION_ALGORITHM_SMOOTH
        ? ctx->stabilization_radius
        : 0;
    nb_buffered_frames += ctx->stabilization_horizon;
    return !ctx->input_eof && ctx->nb_frames_in_progress < nb_buffered_frames
        + 1 + EXTRA_IN_PROGRESS_FRAMES;
}

static int send_output_frame(AVFilterContext *avctx, FrameJob * job) {
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *outlink = avctx->outputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    AVFrame *output_frame = NULL;
    int err;

    output_frame = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (output_frame == NULL) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output_frame, job->input_frame);
    if (err) {
        goto fail;
    }

    err = copy_buffer_to_frame(avctx, job->output_buffer, output_frame);
    if (err) {
        goto fail;
    }

    av_log(
        avctx,
        AV_LOG_VERBOSE,
        "Sending output frame %ld (%d in progress)\n",
        job->num,
        ctx->nb_frames_in_progress
    );
    ctx->nb_frames_in_progress -= 1;

    err = ff_filter_frame(outlink, output_frame);
    if (err < 0) {
        goto fail;
    }

    if (input_frame_wanted(ctx)) {
        ff_inlink_request_frame(inlink);
    }

    if (ctx->input_eof && ctx->nb_frames_in_progress == 0) {
        av_log(avctx, AV_LOG_VERBOSE, "Output reached EOF\n");
        send_eof_to_dewobble_thread(avctx);
        ff_outlink_set_status(outlink, AVERROR_EOF, job->pts);
    }
    return 0;

fail:
    av_log(avctx, AV_LOG_ERROR, "Failed to send output frame: %d\n", err);
    av_frame_free(&output_frame);
    return err;
}

static int try_send_output_frame(AVFilterContext *avctx) {
    int err;
    DewobbleOpenCLContext *ctx = avctx->priv;
    DewobbleMessage *message = (DewobbleMessage *) ff_safe_queue_pop_front(ctx->output_queue);

    if (message == NULL) {
        return 0;
    }

    if (message->type == DEWOBBLE_MESSAGE_TYPE_ERROR) {
        av_log(avctx, AV_LOG_ERROR, "Received error message from dewobble thread\n");
        dewobble_message_free(&message);
        return -1;
    }

    err = send_output_frame(avctx, message->job);
    frame_job_free(avctx, &message->job);
    dewobble_message_free(&message);
    if (err) {
        return err;
    }
    return 0;
}

static int try_consume_input_frame(AVFilterContext *avctx) {
    AVFilterLink *inlink = avctx->inputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    int err = 0;
    AVFrame *input_frame;

    // If necessary, attempt to consume a frame from the input
    if (input_frame_wanted(ctx)) {
        err = ff_inlink_consume_frame(inlink, &input_frame);
        if (err < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to read input frame\n");
            return err;
        } else if (err > 0) {
            av_log(
                avctx,
                AV_LOG_VERBOSE,
                "Consuming input frame %ld (%d in progress)\n",
                ctx->nb_frames_consumed,
                ctx->nb_frames_in_progress
            );
            err = consume_input_frame(avctx, input_frame);

            if (err) {
                av_log(avctx, AV_LOG_ERROR, "Failed to consume input frame: %d\n", err);
                return err;
            }

            // Request more frames if necessary
            if (input_frame_wanted(ctx)) {
                ff_inlink_request_frame(inlink);
            }
        }
    }
    return err;
}

static void check_for_input_eof(AVFilterContext *avctx) {
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *outlink = avctx->outputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    int64_t pts;
    int status;

    // Check for end of input
    if (!ctx->input_eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            av_log(avctx, AV_LOG_VERBOSE, "Reached input EOF\n");
            ctx->input_eof = 1;
            send_eof_to_dewobble_thread(avctx);
            if (ctx->nb_frames_in_progress == 0) {
                ff_outlink_set_status(outlink, AVERROR_EOF, pts);
            }
        } else if (status) {
            av_log(avctx, AV_LOG_ERROR, "INPUT STATUS: %d\n", status);
        }
    }
}

static int activate(AVFilterContext *avctx)
{
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *outlink = avctx->outputs[0];
    int err = 0;

    err = ff_outlink_get_status(outlink);
    if (err) {
        av_log(avctx, AV_LOG_VERBOSE, "forwarding status to inlink: %d\n", err);
        ff_inlink_set_status(inlink, err);
        stop_dewobble_thread_on_error(avctx);
        return 0;
    }

    err = try_consume_input_frame(avctx);
    if (err) {
        av_log(avctx, AV_LOG_ERROR, "try_consume_input_frame failed: %d\n", err);
        goto fail;
    }

    check_for_input_eof(avctx);

    err = try_send_output_frame(avctx);
    if (err) {
        av_log(avctx, AV_LOG_ERROR, "try_send_output_frame failed: %d\n", err);
        goto fail;
    }

    return FFERROR_NOT_READY;

fail:
    stop_dewobble_thread_on_error(avctx);
    ff_outlink_set_status(outlink, AVERROR_UNKNOWN, 0);
    return err;
}


#define OFFSET(x) offsetof(DewobbleOpenCLContext, x)
#define OFFSET_CAMERA(x) offsetof(Camera, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption dewobble_opencl_options[] = {
    // Input camera options
    {
        "in_p",
        "input camera projection model",
        OFFSET(input_camera) + OFFSET_CAMERA(model),
        AV_OPT_TYPE_INT,
        { .i64 = DEWOBBLE_PROJECTION_EQUIDISTANT_FISHEYE },
        0,
        DEWOBBLE_NB_PROJECTIONS - 1,
        FLAGS,
        "model",
    },
    {
        "in_fl",
        "input camera focal length in pixels",
        OFFSET(input_camera) + OFFSET_CAMERA(focal_length),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = 0 },
        0,
        DBL_MAX,
        .flags=FLAGS,
    },
    {
        "in_fx",
        "horizontal coordinate of focal point in input camera (default: center)",
        OFFSET(input_camera) + OFFSET_CAMERA(focal_point_x),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = DBL_MAX },
        -DBL_MAX,
        DBL_MAX,
        .flags=FLAGS,
    },
    {
        "in_fy",
        "vertical coordinate of focal point in input camera (default: center)",
        OFFSET(input_camera) + OFFSET_CAMERA(focal_point_y),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = DBL_MAX },
        -DBL_MAX,
        DBL_MAX,
        .flags=FLAGS,
    },

    // Output camera options
    {
        "out_p",
        "output camera projection model",
        OFFSET(output_camera) + OFFSET_CAMERA(model),
        AV_OPT_TYPE_INT,
        { .i64 = DEWOBBLE_PROJECTION_RECTILINEAR },
        0,
        DEWOBBLE_NB_PROJECTIONS - 1,
        FLAGS,
        "model",
    },
    {
        "out_fl",
        "output camera focal length in pixels",
        OFFSET(output_camera) + OFFSET_CAMERA(focal_length),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = 0 },
        0,
        DBL_MAX,
        .flags=FLAGS,
    },
    {
        "out_w",
        "output camera width in pixels (default: same as input)",
        OFFSET(output_camera) + OFFSET_CAMERA(width),
        AV_OPT_TYPE_INT,
        { .i64 = 0 },
        0,
        SHRT_MAX,
        .flags=FLAGS,
    },
    {
        "out_h",
        "output camera height in pixels (default: same as input)",
        OFFSET(output_camera) + OFFSET_CAMERA(height),
        AV_OPT_TYPE_INT,
        { .i64 = 0 },
        0,
        SHRT_MAX,
        .flags=FLAGS,
    },
    {
        "out_fx",
        "horizontal coordinate of focal point in output camera (default: center)",
        OFFSET(output_camera) + OFFSET_CAMERA(focal_point_x),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = DBL_MAX },
        -DBL_MAX,
        DBL_MAX,
        .flags=FLAGS,
    },
    {
        "out_fy",
        "vertical coordinate of focal point in output camera (default: center)",
        OFFSET(output_camera) + OFFSET_CAMERA(focal_point_y),
        AV_OPT_TYPE_DOUBLE,
        { .dbl = DBL_MAX },
        -DBL_MAX,
        DBL_MAX,
        .flags=FLAGS,
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
        "stab_r",
        "for Savitzky-Golay smoothing: the number of frames to look ahead and behind",
        OFFSET(stabilization_radius),
        AV_OPT_TYPE_INT,
        { .i64 = 15 },
        1,
        INT_MAX,
        FLAGS,
    },
    {
        "stab_h",
        "for stabilization: the number of frames to look ahead to interpolate rotation in frames where it cannot be detected",
        OFFSET(stabilization_horizon),
        AV_OPT_TYPE_INT,
        { .i64 = 30 },
        0,
        INT_MAX,
        FLAGS,
    },

    // General options
    {
        "interp",
        "interpolation algorithm",
        OFFSET(interpolation_algorithm),
        AV_OPT_TYPE_INT,
        { .i64 = DEWOBBLE_INTERPOLATION_LINEAR },
        0,
        DEWOBBLE_NB_INTERPOLATIONS - 1,
        FLAGS,
        "interpolation",
    },

    // Camera models
    {
        "rect",
        "rectilinear projection",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_PROJECTION_RECTILINEAR },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "model"
    },
    {
        "fish",
        "equidistant fisheye projection",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_PROJECTION_EQUIDISTANT_FISHEYE },
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
        "none",
        "do not apply stabilization",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = STABILIZATION_ALGORITHM_ORIGINAL },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "stab"
    },
    {
        "sg",
        "smooth the camera orientation using a Savitzky-Golay filter",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = STABILIZATION_ALGORITHM_SMOOTH },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "stab"
    },

    // Interpolation algorithms
    {
        "nearest",
        "nearest neighbour interpolation (fast)",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_INTERPOLATION_NEAREST },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "interpolation"
    },
    {
        "linear",
        "bilinear interpolation (fast)",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_INTERPOLATION_LINEAR },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "interpolation"
    },
    {
        "cubic",
        "bicubic interpolation (medium)",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_INTERPOLATION_CUBIC },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "interpolation"
    },
    {
        "lanczos",
        "Lanczos4, in an 8x8 neighbourhood (slow)",
        0,
        AV_OPT_TYPE_CONST,
        { .i64 = DEWOBBLE_INTERPOLATION_LANCZOS4 },
        INT_MIN,
        INT_MAX,
        FLAGS,
        "interpolation"
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
