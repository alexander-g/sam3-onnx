#!/usr/bin/env python3

import argparse
import pathlib
import typing

import imgviz
import numpy as np
import onnxruntime
import PIL.Image
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        default=pathlib.Path("images/bus.jpg"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help="Text prompt for segmentation.",
    )
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    sess_image = onnxruntime.InferenceSession("models/sam3_image_encoder.onnx")
    sess_language = onnxruntime.InferenceSession("models/sam3_language_encoder.onnx")
    sess_decode = onnxruntime.InferenceSession("models/sam3_decoder.onnx")

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")
    logger.debug("original image size: {}", image.size)

    logger.debug("running image encoder...")
    output = sess_image.run(
        None, {"image": np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1)}
    )
    assert len(output) == 6
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]
    logger.debug("finished running image encoder")

    logger.debug("running language encoder...")
    output = sess_language.run(
        None, {"tokenized": tokenize(texts=[args.prompt], context_length=32)}
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    language_mask: NDArray = output[0]
    language_features: NDArray = output[1]
    language_embeds: NDArray = output[2]
    logger.debug("finished running language encoder")

    logger.debug("running decoder...")
    output = sess_decode.run(
        None,
        {
            "original_height": np.array([image.height], dtype=np.int64),
            "original_width": np.array([image.width], dtype=np.int64),
            "backbone_fpn_0": backbone_fpn[0],
            "backbone_fpn_1": backbone_fpn[1],
            "backbone_fpn_2": backbone_fpn[2],
            "vision_pos_enc_0": vision_pos_enc[0],
            "vision_pos_enc_1": vision_pos_enc[1],
            "vision_pos_enc_2": vision_pos_enc[2],
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
        },
    )
    assert len(output) == 3
    assert all(isinstance(o, np.ndarray) for o in output)
    output = typing.cast(list[NDArray], output)
    boxes: NDArray = output[0]
    scores: NDArray = output[1]
    masks: NDArray = output[2]
    logger.debug("finished running decoder")

    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{args.prompt}: {s:.0%}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
