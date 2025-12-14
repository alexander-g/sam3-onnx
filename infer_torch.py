#!/usr/bin/env python3

import argparse
import pathlib
import sys

import cv2
import imgviz
import numpy as np
import PIL.Image
import torch
from loguru import logger


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
        help="Path to the input image.",
        required=True,
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt for segmentation.",
    )
    prompt_group.add_argument(
        "--box-prompt",
        type=str,
        nargs="?",
        const="0,0,0,0",
        help="Box prompt for segmentation in format: cx cy w h (normalized).",
    )
    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    if args.box_prompt:
        args.box_prompt = [float(x) for x in args.box_prompt.split(",")]
        if len(args.box_prompt) != 4:
            logger.error("box_prompt must have 4 values: cx, cy, w, h")
            sys.exit(1)

    image: PIL.Image.Image = PIL.Image.open(args.image).convert("RGB")

    if args.box_prompt == [0, 0, 0, 0]:
        logger.info("please select box prompt in the image window")
        x, y, w, h = cv2.selectROI(
            "Select box prompt and press ENTER or SPACE",
            np.asarray(image)[:, :, ::-1],
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        if [x, y, w, h] == [0, 0, 0, 0]:
            logger.warning("no box prompt selected, exiting")
            sys.exit(1)

        args.box_prompt = [
            (x + w / 2) / image.width,
            (y + h / 2) / image.height,
            w / image.width,
            h / image.height,
        ]
        logger.debug("box_prompt: {!r}", ",".join(f"{x:.3f}" for x in args.box_prompt))

    # XXX: those imports has to be after cv2.selectROI to avoid segfault
    from sam3.model.sam3_image import Sam3Image  # type: ignore[unresolved-import]
    from sam3.model.sam3_image_processor import (  # type: ignore[unresolved-import]
        Sam3Processor,
    )
    from sam3.model_builder import (  # type: ignore[unresolved-import]
        build_sam3_image_model,
    )

    model: Sam3Image = build_sam3_image_model()

    if 0:
        with torch.no_grad():
            get_replace_freqs_cis(model)

    processor: Sam3Processor = Sam3Processor(model)
    state = processor.set_image(image)
    if args.text_prompt:
        output = processor.set_text_prompt(prompt=args.text_prompt, state=state)
    elif args.box_prompt:
        output = processor.add_geometric_prompt(
            box=args.box_prompt,
            label=True,
            state=state,
        )

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    logger.debug(
        "output: {}",
        {"masks": masks.shape, "boxes": boxes.shape, "scores": scores.shape},
    )

    text_prompt: str = args.text_prompt if args.text_prompt else "visual"
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks.cpu().numpy()[:, 0, :, :],
        bboxes=boxes.cpu().numpy()[:, [1, 0, 3, 2]],
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
    )
    imgviz.io.pil_imshow(viz)


if __name__ == "__main__":
    main()
