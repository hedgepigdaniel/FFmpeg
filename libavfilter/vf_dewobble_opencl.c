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

#include "libavutil/avassert.h"
#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "internal.h"
#include "opencl.h"
#include "opencl_source.h"
#include "video.h"
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
    int initialised;
    Camera input_camera;
    Camera output_camera;
    StabilizationAlgorithm stabilization_algorithm;
    int stabilization_radius;
} DewobbleOpenCLContext;

static int dewobble_opencl_init(AVFilterContext *avctx)
{
    DewobbleOpenCLContext *ctx = avctx->priv;
    ctx->initialised = 1;
    return 0;
}

static int dewobble_opencl_config_output(AVFilterLink *outlink)
{
    AVFilterContext *avctx = outlink->src;
    DewobbleOpenCLContext *s = avctx->priv;
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


static int dewobble_opencl_filter_frame(AVFilterLink *inlink, AVFrame *input)
{
    AVFilterContext    *avctx = inlink->dst;
    AVFilterLink     *outlink = avctx->outputs[0];
    DewobbleOpenCLContext *ctx = avctx->priv;
    AVFrame *output = NULL;
    int err;

    av_log(ctx, AV_LOG_DEBUG, "Filter input: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(input->format),
           input->width, input->height, input->pts);

    if (!input->hw_frames_ctx)
        return AVERROR(EINVAL);

    if (!ctx->initialised) {
        err = dewobble_opencl_init(avctx);
        if (err < 0)
            goto fail;
    }

    output = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!output) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    err = av_frame_copy_props(output, input);
    if (err < 0)
        goto fail;

    av_frame_free(&input);

    av_log(ctx, AV_LOG_DEBUG, "Filter output: %s, %ux%u (%"PRId64").\n",
           av_get_pix_fmt_name(output->format),
           output->width, output->height, output->pts);

    return ff_filter_frame(outlink, output);

fail:
    av_frame_free(&input);
    av_frame_free(&output);
    return err;
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
        .filter_frame = &dewobble_opencl_filter_frame,
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
    .init           = &ff_opencl_filter_init,
    .uninit         = &ff_opencl_filter_uninit,
    .query_formats  = &ff_opencl_filter_query_formats,
    .inputs         = dewobble_opencl_inputs,
    .outputs        = dewobble_opencl_outputs,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
