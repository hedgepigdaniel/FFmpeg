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

#include "avfilter.h"
#include "filters.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"
#include "safe_queue.h"
#include "transpose.h"

typedef enum CameraModel {
    CAMERA_MODEL_RECTILINEAR,
    CAMERA_MODEL_EQUIDISTANT_FISHEYE,
    NB_CAMERA_MODELS,
} CameraModel;

typedef struct Camera {
    CameraModel model;
    double focal_length;
} Camera;

typedef enum StabilizationAlgorithm {
    STABILIZATION_ALGORITHM_ORIGINAL,
    STABILIZATION_ALGORITHM_FIXED,
    STABILIZATION_ALGORITHM_SMOOTH,
    NB_STABILIZATION_ALGORITHMS,
} StabilizationAlgorithm;

typedef struct DewobbleOpenCLContext {
    OpenCLFilterContext ocf;
    cl_command_queue command_queue;

    // Options
    Camera input_camera;
    Camera output_camera;
    StabilizationAlgorithm stabilization_algorithm;
    int stabilization_radius;

    // State
    int initialized;
    int input_eof;
    int nb_frames_in_progress;
    int dewobble_thread_created;
    pthread_t dewobble_thread;
    SafeQueue *input_frame_queue;
    SafeQueue *output_frame_queue;
} DewobbleOpenCLContext;

typedef struct QueuedFrame {
    int err;
    int64_t pts;
    AVFrame *input;
    AVFrame *output;
} QueuedFrame;

#define EXTRA_IN_PROGRESS_FRAMES 2

static void *dewobble_thread(void *arg) {
    AVFilterContext *avctx = arg;
    DewobbleOpenCLContext *ctx = avctx->priv;
    QueuedFrame *queued_frame = NULL;

    av_log(avctx, AV_LOG_VERBOSE, "Started worker thread\n");
    while (1) {
        queued_frame = ff_safe_queue_pop_front_blocking(ctx->input_frame_queue);

        ff_safe_queue_push_back(ctx->output_frame_queue, queued_frame);
        av_log(avctx, AV_LOG_VERBOSE, "Worker thread: pushed frame.\n");

        if (queued_frame->err == AVERROR_EOF) {
            break;
        }
    }
    av_log(avctx, AV_LOG_VERBOSE, "Worker thread: reached EOF, exiting.\n");
    return NULL;
}

static int dewobble_opencl_init(AVFilterContext *avctx) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    ctx->input_frame_queue = ff_safe_queue_create();
    if (ctx->input_frame_queue == NULL) {
        return AVERROR(ENOMEM);
    }
    ctx->output_frame_queue = ff_safe_queue_create();
    if (ctx->output_frame_queue == NULL) {
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
    ff_safe_queue_destroy(ctx->input_frame_queue);
    ff_safe_queue_destroy(ctx->output_frame_queue);
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

    ctx->command_queue = clCreateCommandQueue(ctx->ocf.hwctx->context,
                                              ctx->ocf.hwctx->device_id,
                                              0, &cle);
    if (cle) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create OpenCL command queue %d.\n", cle);
        return AVERROR(EIO);
    }

    pthread_create(&ctx->dewobble_thread, NULL, dewobble_thread, avctx);
    ctx->dewobble_thread_created = 1;

    ctx->initialized = 1;
    return 0;
}

static int dewobble_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    AVFilterLink *inlink = avctx->inputs[0];
    const AVPixFmtDescriptor *desc_in  = av_pix_fmt_desc_get(inlink->format);
    int ret;


    if (desc_in->log2_chroma_w != desc_in->log2_chroma_h) {
        av_log(avctx, AV_LOG_ERROR, "Input format %s not supported.\n",
               desc_in->name);
        return AVERROR(EINVAL);
    }

    ret = ff_opencl_filter_config_output(outlink);
    if (ret < 0)
        return ret;

    av_log(avctx, AV_LOG_VERBOSE, "w:%d h:%d\n", inlink->w, inlink->h);
    return 0;
}

static QueuedFrame *queued_frame_create(int err, int64_t pts, AVFrame *input, AVFrame *output) {
    QueuedFrame *result = av_mallocz(sizeof(QueuedFrame));
    if (result == NULL) {
        return NULL;
    }
    result->err = err;
    result->pts = pts;
    result->input = input;
    result->output = output;
    return result;
}

static void queued_frame_free(QueuedFrame **frame) {
    if (*frame != NULL) {
        av_frame_free(&(*frame)->input);
        av_frame_free(&(*frame)->output);
    }
    *frame = NULL;
}

static cl_int cl_buffer_from_opencl_frame(
    AVFilterContext *avctx,
    cl_context context,
    cl_command_queue command_queue,
    AVFrame *frame,
    cl_mem *dst_buffer
) {
    cl_mem luma = (cl_mem) frame->data[0];
    cl_mem chroma = (cl_mem) frame->data[1];
    cl_int ret = 0;
    cl_image_format luma_fmt = { 0, 0 };
    cl_image_format chroma_fmt = { 0, 0 };
    size_t luma_w = 0;
    size_t luma_h = 0;
    size_t chroma_w = 0;
    size_t chroma_h = 0;
    size_t src_origin[3] = { 0, 0, 0 };
    size_t luma_region[3] = { 0, 0, 1 };
    size_t chroma_region[3] = { 0, 0, 1 };

    ret = clGetImageInfo(luma, CL_IMAGE_FORMAT, sizeof(cl_image_format), &luma_fmt, 0);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get luma image format: %d\n", ret);
        return ret;
    }
    ret = clGetImageInfo(chroma, CL_IMAGE_FORMAT, sizeof(cl_image_format), &chroma_fmt, 0);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get chroma image format: %d\n", ret);
        return ret;
    }
    if (luma_fmt.image_channel_data_type != CL_UNORM_INT8 ||
        chroma_fmt.image_channel_data_type != CL_UNORM_INT8) {
        av_log(avctx, AV_LOG_ERROR, "Incorrect channel type\n");
        return 1;
    }
    if (luma_fmt.image_channel_order != CL_R ||
        chroma_fmt.image_channel_order != CL_RG) {
        av_log(avctx, AV_LOG_ERROR, "Incorrect channel order\n");
        return 1;
    }

    ret |= clGetImageInfo(luma, CL_IMAGE_WIDTH, sizeof(size_t), &luma_w, 0);
    ret |= clGetImageInfo(luma, CL_IMAGE_HEIGHT, sizeof(size_t), &luma_h, 0);
    ret |= clGetImageInfo(chroma, CL_IMAGE_WIDTH, sizeof(size_t), &chroma_w, 0);
    ret |= clGetImageInfo(chroma, CL_IMAGE_HEIGHT, sizeof(size_t), &chroma_h, 0);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get image dimensions: %d\n", ret);
        return ret;
    }

    if (luma_w != 2 * chroma_w || luma_h != 2 *chroma_h ) {
        av_log(avctx, AV_LOG_ERROR, "Incorrect dimensions\n");
        return 1;
    }

    *dst_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (luma_h + chroma_h) * luma_w, NULL, &ret);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create buffer: %d\n", ret);
        return ret;
    }

    luma_region[0] = luma_w;
    luma_region[1] = luma_h;
    chroma_region[0] = chroma_w;
    chroma_region[1] = chroma_h;

    ret = clEnqueueCopyImageToBuffer(
        command_queue,
        luma,
        *dst_buffer,
        src_origin,
        luma_region,
        0,
        0,
        NULL,
        NULL
    );
    ret |= clEnqueueCopyImageToBuffer(
        command_queue,
        chroma,
        *dst_buffer,
        src_origin,
        chroma_region,
        luma_w * luma_h * 1,
        0,
        NULL,
        NULL
    );
    ret |= clFinish(command_queue);
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to copy images to buffer: %d\n", ret);
        clReleaseMemObject(*dst_buffer);
        *dst_buffer = NULL;
        return ret;
    }

    return ret;
}

static int consume_input_frame(AVFilterContext *avctx, AVFrame *frame) {
    DewobbleOpenCLContext *ctx = avctx->priv;
    AVFilterLink *outlink = avctx->outputs[0];
    QueuedFrame *queued_frame = NULL;
    cl_mem frame_buffer = NULL;
    int ret = 0;

    if (!frame->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!ctx->initialized) {
        av_log(avctx, AV_LOG_VERBOSE, "Initializing\n");
        ret = dewobble_opencl_frames_init(avctx);
        if (ret < 0) {
            return ret;
        }
    }

    ret = cl_buffer_from_opencl_frame(
        avctx,
        ctx->ocf.hwctx->context,
        ctx->command_queue,
        frame,
        &frame_buffer
    );
    if (ret) {
        av_log(avctx, AV_LOG_ERROR, "Failed to map OpenCL frame to OpenCL buffer: %d\n", ret);
        return AVERROR(EINVAL);
    }

    queued_frame = queued_frame_create(
        0,
        frame->pts,
        frame,
        ff_get_video_buffer(outlink, outlink->w, outlink->h)
    );
    if (queued_frame == NULL) {
        return AVERROR(ENOMEM);
    } else if (queued_frame->output == NULL) {
        queued_frame_free(&queued_frame);
        return AVERROR(ENOMEM);
    }

    ret = av_frame_copy_props(queued_frame->output, frame);
    if (ret < 0) {
        return ret;
    }

    ret = ff_safe_queue_push_back(ctx->input_frame_queue, queued_frame);
    if (ret < 0)
    {
        return ret;
    }
    ctx->nb_frames_in_progress += 1;
    return 0;
}

static int input_frame_wanted(DewobbleOpenCLContext *ctx) {
    return !ctx->input_eof && ctx->nb_frames_in_progress < ctx->stabilization_radius
        + 1 + EXTRA_IN_PROGRESS_FRAMES;
}

static int handle_input_eof(DewobbleOpenCLContext *ctx, int64_t pts) {
    QueuedFrame *queued_frame = NULL;
    int ret;

    ctx->input_eof = 1;
    queued_frame = queued_frame_create(AVERROR_EOF, pts, NULL, NULL);
    if (queued_frame == NULL) {
        return AVERROR(ENOMEM);
    }
    ret = ff_safe_queue_push_back(ctx->input_frame_queue, queued_frame);
    if (ret < 0)
    {
        return ret;
    }
    return 0;
}

static int activate(AVFilterContext *avctx)
{
    AVFilterLink *inlink = avctx->inputs[0];
    AVFilterLink *outlink = avctx->outputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    AVFrame *frame = NULL;
    QueuedFrame *queued_output_frame = NULL;
    int64_t pts;
    int ret = 0, status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    // If possible, output a frame
    av_log(avctx, AV_LOG_VERBOSE, "Checking for output frame...\n");
    queued_output_frame = (QueuedFrame *) ff_safe_queue_pop_front(ctx->output_frame_queue);
    if (queued_output_frame != NULL) {
        if (queued_output_frame->err == AVERROR_EOF) {
            // Propagate EOF to output
            av_log(avctx, AV_LOG_VERBOSE, "Output reached EOF\n");
            ff_outlink_set_status(outlink, AVERROR_EOF, queued_output_frame->pts);
        } else if (queued_output_frame->err) {
            ret = queued_output_frame->err;
        } else {
            av_log(
                avctx,
                AV_LOG_VERBOSE,
                "Sending output frame %d -> %d\n",
                ctx->nb_frames_in_progress,
                ctx->nb_frames_in_progress - 1
            );
            ctx->nb_frames_in_progress -= 1;
            ret = ff_filter_frame(outlink, queued_output_frame->output);
            if (ret == 0) {
                queued_output_frame->output = NULL;
            }
        }
        queued_frame_free(&queued_output_frame);
        return ret;
    } else if (ctx->input_eof) {
        // No more input frames will cause the filter to wake up,
        // so keep checking until the EOF message arrives
        ff_filter_set_ready(avctx, 1);
    }

    // If necessary, attempt to consume a frame from the input
    if (input_frame_wanted(ctx)) {
        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to consume input frame\n");
            return ret;
        } else if (ret > 0) {
            av_log(avctx, AV_LOG_VERBOSE, "Consuming input frame %d -> %d\n", ctx->nb_frames_in_progress, ctx->nb_frames_in_progress + 1);
            ret = consume_input_frame(avctx, frame);
            if (ret) {
                av_frame_free(&frame);
                return ret;
            }
        } else {
            av_log(avctx, AV_LOG_VERBOSE, "No input frame available\n");
        }

        // If we still need more input frames, request them now
        if (input_frame_wanted(ctx)) {
            av_log(avctx, AV_LOG_VERBOSE, "Requesting input frame\n");
            ff_inlink_request_frame(inlink);
        }
    }

    // Check for end of input
    if (!ctx->input_eof && ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            av_log(avctx, AV_LOG_VERBOSE, "Reached end of input\n");
            ret = handle_input_eof(ctx, pts);
            if (ret) {
                return ret;
            }
            ff_filter_set_ready(avctx, 1);
        }
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
        .config_props = &ff_opencl_filter_config_input,
    },
    { NULL }
};

static const AVFilterPad dewobble_opencl_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = &dewobble_opencl_config_output,
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
