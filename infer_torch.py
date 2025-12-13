#!/usr/bin/env python3

import argparse
import pathlib

import imgviz
import numpy as np
import PIL.Image
import torch
from loguru import logger

from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
    Sam3Processor,
)
from sam3.model_builder import build_sam3_image_model  # type: ignore[unresolved-import]


def get_replace_freqs_cis(module):
    if hasattr(module, "freqs_cis"):
        freqs_cos = module.freqs_cis.real.float()
        freqs_sin = module.freqs_cis.imag.float()
        # Replace the buffer
        module.register_buffer("freqs_cos", freqs_cos)
        module.register_buffer("freqs_sin", freqs_sin)
        del module.freqs_cis  # Remove complex version
    for child in module.children():
        get_replace_freqs_cis(child)


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

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")

    model: Sam3Image = build_sam3_image_model()

    if 0:
        with torch.no_grad():
            get_replace_freqs_cis(model)

    processor: Sam3Processor = Sam3Processor(model)
    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt=args.prompt, state=state)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks.cpu().numpy()[:, 0, :, :],
        bboxes=boxes.cpu().numpy()[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{s:.2f}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
