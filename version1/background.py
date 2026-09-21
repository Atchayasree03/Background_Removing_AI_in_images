#!/usr/bin/env python3

"""
============================================================
WAREHOUSE OBJECT REMOVAL - VERSION 7
============================================================

Pipeline:

    Input Image
         |
         v
    GroundingDINO
         |
         v
    Object Bounding Boxes
         |
         v
       SAM2
         |
         v
    Object Segmentation
         |
         v
    Mask Refinement
         |
         v
    Floor-Aware Reconstruction
         |
         v
    Multi-Scale OpenCV Inpainting
         |
         v
    LaMa for Remaining Difficult Areas
         |
         v
    Floor-Line Restoration
         |
         v
    Clean Background

Usage:

    python3 background.py warehouse.jpeg

Outputs:

    clean_background.png
    object_mask.png
    detections.png
    reconstruction_mask.png
    opencv_background.png
    final_background.png

============================================================
"""

# ============================================================
# ENVIRONMENT
# ============================================================

import os
import sys
from pathlib import Path

# CPU ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# YOUR EXISTING PATHS
# ============================================================

GROUNDINGDINO_DIR = (
    BASE_DIR / "GroundingDINO"
)

SAM2_DIR = (
    Path.home() / "sam2"
)


# Python import paths

sys.path.insert(
    0,
    str(GROUNDINGDINO_DIR),
)

sys.path.insert(
    0,
    str(SAM2_DIR),
)


# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import torch

from PIL import Image

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

from sam2.build_sam import build_sam2

from sam2.sam2_image_predictor import (
    SAM2ImagePredictor,
)

from simple_lama_inpainting import (
    SimpleLama,
)


# ============================================================
# MODEL PATHS
# ============================================================

GROUNDING_CONFIG = (
    GROUNDINGDINO_DIR
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinT_OGC.py"
)


GROUNDING_CHECKPOINT = (
    GROUNDINGDINO_DIR
    / "weights"
    / "groundingdino_swint_ogc.pth"
)


SAM2_CONFIG = (
    "sam2.1/sam2.1_hiera_s.yaml"
)


SAM2_CHECKPOINT = (
    SAM2_DIR
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)


# ============================================================
# OBJECT PROMPT
# ============================================================

TEXT_PROMPT = (
    "pallet . "
    "wooden pallet . "
    "box . "
    "cat ."
    "cardboard box . "
    "container . "
    "plastic container . "
    "storage container . "
    "package . "
    "carton . "
    "crate . "
)


# ============================================================
# DETECTION PARAMETERS
# ============================================================

BOX_THRESHOLD = 0.20

TEXT_THRESHOLD = 0.15

MIN_OBJECT_AREA = 800


# ============================================================
# MASK PARAMETERS
# ============================================================

MASK_DILATION = 3

MASK_CLOSE_SIZE = 7

MASK_CLOSE_ITERATIONS = 1


# ============================================================
# RECONSTRUCTION PARAMETERS
# ============================================================

# OpenCV first pass.

TELEA_RADIUS = 5

NS_RADIUS = 5


# Multi-scale reconstruction.

SMALL_SCALE = 0.50

VERY_SMALL_SCALE = 0.25


# LaMa is used after OpenCV.

USE_LAMA = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cpu"


print()
print("========================================")
print("WAREHOUSE BACKGROUND REMOVAL V7")
print("========================================")

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "CUDA available:",
    torch.cuda.is_available(),
)

print(
    "Device:",
    DEVICE,
)

print("========================================")


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path):

    path = Path(path)

    if not path.exists():

        print()
        print("ERROR")
        print("----------------------------------------")
        print("File not found:")
        print(path)
        print("----------------------------------------")
        print()

        sys.exit(1)


# ============================================================
# CHECK MODELS
# ============================================================

def check_model_files():

    print()
    print("========================================")
    print("CHECKING MODEL FILES")
    print("========================================")


    print()
    print("GroundingDINO config:")
    print(GROUNDING_CONFIG)

    check_file(
        GROUNDING_CONFIG
    )


    print()
    print("GroundingDINO checkpoint:")
    print(GROUNDING_CHECKPOINT)

    check_file(
        GROUNDING_CHECKPOINT
    )


    sam2_config_path = (
        SAM2_DIR
        / "sam2"
        / "configs"
        / "sam2.1"
        / "sam2.1_hiera_s.yaml"
    )


    print()
    print("SAM2 config:")
    print(sam2_config_path)

    check_file(
        sam2_config_path
    )


    print()
    print("SAM2 checkpoint:")
    print(SAM2_CHECKPOINT)

    check_file(
        SAM2_CHECKPOINT
    )


    print()
    print("All model files found.")


# ============================================================
# LOAD GROUNDINGDINO
# ============================================================

def load_grounding_dino():

    print()
    print("========================================")
    print("LOADING GROUNDINGDINO")
    print("========================================")


    model = load_model(
        str(GROUNDING_CONFIG),
        str(GROUNDING_CHECKPOINT),
        device=DEVICE,
    )


    print(
        "GroundingDINO loaded."
    )


    return model


# ============================================================
# LOAD SAM2
# ============================================================

def load_sam2():

    print()
    print("========================================")
    print("LOADING SAM2")
    print("========================================")


    model = build_sam2(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=DEVICE,
    )


    predictor = SAM2ImagePredictor(
        model
    )


    print(
        "SAM2 loaded."
    )


    return predictor


# ============================================================
# DETECTION
# ============================================================

def detect_objects(
    grounding_model,
    image_path,
):

    print()
    print("========================================")
    print("GROUNDINGDINO DETECTION")
    print("========================================")


    image_source, image = (
        load_image(
            str(image_path)
        )
    )


    boxes, logits, phrases = (
        predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )
    )


    print()
    print(
        "Raw detections:",
        len(boxes),
    )


    return (
        image_source,
        boxes,
        logits,
        phrases,
    )


# ============================================================
# CONVERT BOXES
# ============================================================

def convert_boxes(
    boxes,
    logits,
    phrases,
    width,
    height,
):

    scale = torch.tensor(
        [
            width,
            height,
            width,
            height,
        ],
        dtype=torch.float32,
    )


    scaled = (
        boxes.detach().cpu()
        * scale
    )


    detections = []


    for index, box in enumerate(
        scaled
    ):

        cx, cy, bw, bh = (
            box.tolist()
        )


        x1 = int(
            cx - bw / 2
        )

        y1 = int(
            cy - bh / 2
        )

        x2 = int(
            cx + bw / 2
        )

        y2 = int(
            cy + bh / 2
        )


        x1 = max(
            0,
            min(
                x1,
                width - 1,
            ),
        )


        y1 = max(
            0,
            min(
                y1,
                height - 1,
            ),
        )


        x2 = max(
            0,
            min(
                x2,
                width - 1,
            ),
        )


        y2 = max(
            0,
            min(
                y2,
                height - 1,
            ),
        )


        area = (
            max(
                0,
                x2 - x1,
            )
            *
            max(
                0,
                y2 - y1,
            )
        )


        if (
            area < MIN_OBJECT_AREA
            or x2 <= x1
            or y2 <= y1
        ):

            continue


        detections.append(
            {
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "score": float(
                    logits[index]
                ),
                "phrase": (
                    phrases[index]
                    if index < len(phrases)
                    else "object"
                ),
            }
        )


    return detections


# ============================================================
# REMOVE DUPLICATE / HEAVILY OVERLAPPING BOXES
# ============================================================

def calculate_iou(
    box1,
    box2,
):

    x1 = max(
        box1[0],
        box2[0],
    )

    y1 = max(
        box1[1],
        box2[1],
    )

    x2 = min(
        box1[2],
        box2[2],
    )

    y2 = min(
        box1[3],
        box2[3],
    )


    intersection_width = max(
        0,
        x2 - x1,
    )


    intersection_height = max(
        0,
        y2 - y1,
    )


    intersection = (
        intersection_width
        *
        intersection_height
    )


    area1 = (
        max(
            0,
            box1[2] - box1[0],
        )
        *
        max(
            0,
            box1[3] - box1[1],
        )
    )


    area2 = (
        max(
            0,
            box2[2] - box2[0],
        )
        *
        max(
            0,
            box2[3] - box2[1],
        )
    )


    union = (
        area1
        +
        area2
        -
        intersection
    )


    if union <= 0:

        return 0.0


    return (
        intersection / union
    )


def remove_duplicate_boxes(
    detections,
):

    print()
    print("Removing duplicate detections...")


    # Highest confidence first.

    detections = sorted(
        detections,
        key=lambda d: d["score"],
        reverse=True,
    )


    selected = []


    for detection in detections:

        current_box = (
            detection["box"]
        )


        keep = True


        for existing in selected:

            iou = calculate_iou(
                current_box,
                existing["box"],
            )


            if iou > 0.70:

                keep = False

                break


        if keep:

            selected.append(
                detection
            )


    print(
        "Before:",
        len(detections),
    )

    print(
        "After:",
        len(selected),
    )


    return selected


# ============================================================
# SAM2 MASK
# ============================================================

def create_sam_mask(
    predictor,
    image_source,
    detections,
):

    print()
    print("========================================")
    print("SAM2 SEGMENTATION")
    print("========================================")


    image_rgb = cv2.cvtColor(
        image_source,
        cv2.COLOR_BGR2RGB,
    )


    height, width = (
        image_rgb.shape[:2]
    )


    final_mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )


    predictor.set_image(
        image_rgb
    )


    for index, detection in enumerate(
        detections
    ):

        print(
            f"Segmenting "
            f"{index + 1}/"
            f"{len(detections)}"
        )


        box = detection[
            "box"
        ]


        box_np = np.array(
            box,
            dtype=np.float32,
        )


        try:

            with torch.inference_mode():

                masks, scores, _ = (
                    predictor.predict(
                        box=box_np,
                        multimask_output=True,
                    )
                )


        except Exception as exc:

            print(
                "SAM2 error:",
                exc,
            )

            continue


        if (
            masks is None
            or len(masks) == 0
        ):

            continue


        best_index = int(
            np.argmax(scores)
        )


        selected_mask = (
            masks[
                best_index
            ]
        )


        final_mask[
            selected_mask > 0
        ] = 255


    return final_mask


# ============================================================
# REFINE MASK
# ============================================================

def refine_mask(
    mask,
):

    print()
    print("========================================")
    print("REFINING OBJECT MASK")
    print("========================================")


    result = mask.copy()


    # --------------------------------------------------------
    # Remove tiny noise.
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MASK_CLOSE_SIZE,
            MASK_CLOSE_SIZE,
        ),
    )


    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MASK_CLOSE_ITERATIONS,
    )


    # --------------------------------------------------------
    # Slight dilation.
    # --------------------------------------------------------

    if MASK_DILATION > 0:

        dilation_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    MASK_DILATION,
                    MASK_DILATION,
                ),
            )
        )


        result = cv2.dilate(
            result,
            dilation_kernel,
            iterations=1,
        )


    # --------------------------------------------------------
    # Remove very tiny connected components.
    # --------------------------------------------------------

    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            result,
            connectivity=8,
        )
    )


    cleaned = np.zeros_like(
        result
    )


    for label in range(
        1,
        number_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]


        if area >= 500:

            cleaned[
                labels == label
            ] = 255


    result = cleaned


    return result


# ============================================================
# SAVE DETECTION IMAGE
# ============================================================

def save_detection_preview(
    image,
    detections,
    output_path,
):

    preview = image.copy()


    for detection in detections:

        x1, y1, x2, y2 = (
            detection["box"]
        )


        cv2.rectangle(
            preview,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )


        label = (
            f'{detection["phrase"]} '
            f'{detection["score"]:.2f}'
        )


        cv2.putText(
            preview,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


    cv2.imwrite(
        str(output_path),
        preview,
    )


# ============================================================
# CREATE FLOOR CONFIDENCE MAP
# ============================================================

def create_floor_confidence(
    image,
    object_mask,
):

    print()
    print("Creating floor confidence map...")


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )


    # --------------------------------------------------------
    # Your warehouse floor is mainly green/yellow/cyan.
    #
    # We use broad ranges instead of a fixed exact color.
    # --------------------------------------------------------

    h = hsv[:, :, 0]

    s = hsv[:, :, 1]

    v = hsv[:, :, 2]


    floor1 = (
        (h >= 20)
        &
        (h <= 95)
        &
        (s >= 25)
        &
        (v >= 45)
    )


    # Less restrictive fallback based on saturation/value.

    floor2 = (
        (s >= 20)
        &
        (v >= 45)
    )


    floor = (
        floor1 | floor2
    ).astype(
        np.uint8
    ) * 255


    # Objects cannot be source floor.

    floor[
        object_mask > 0
    ] = 0


    # Remove very small regions.

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            9,
            9,
        ),
    )


    floor = cv2.morphologyEx(
        floor,
        cv2.MORPH_OPEN,
        kernel,
    )


    floor = cv2.morphologyEx(
        floor,
        cv2.MORPH_CLOSE,
        kernel,
    )


    return floor


# ============================================================
# MULTI-SCALE OPENCV INPAINTING
# ============================================================

def multiscale_inpaint(
    image,
    mask,
):

    print()
    print("========================================")
    print("MULTI-SCALE FLOOR RECONSTRUCTION")
    print("========================================")


    original = image.copy()


    # --------------------------------------------------------
    # We don't want to inpaint the entire image blindly.
    #
    # Only object regions are holes.
    # --------------------------------------------------------

    clean_mask = mask.copy()


    # --------------------------------------------------------
    # PASS 1
    #
    # Downsample.
    #
    # Large structures become easier for the inpainting
    # algorithm to bridge.
    # --------------------------------------------------------

    small_width = int(
        image.shape[1]
        * SMALL_SCALE
    )


    small_height = int(
        image.shape[0]
        * SMALL_SCALE
    )


    small_image = cv2.resize(
        image,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


    small_mask = cv2.resize(
        clean_mask,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )


    print(
        "Small-scale inpainting..."
    )


    small_result = cv2.inpaint(
        small_image,
        small_mask,
        TELEA_RADIUS,
        cv2.INPAINT_TELEA,
    )


    # --------------------------------------------------------
    # Resize back.
    # --------------------------------------------------------

    result = cv2.resize(
        small_result,
        (
            image.shape[1],
            image.shape[0],
        ),
        interpolation=cv2.INTER_CUBIC,
    )


    # --------------------------------------------------------
    # PASS 2
    #
    # Blend only object regions.
    # --------------------------------------------------------

    result[
        clean_mask == 0
    ] = original[
        clean_mask == 0
    ]


    # --------------------------------------------------------
    # Full-resolution Navier-Stokes pass.
    #
    # This is useful for extending floor gradients and
    # large smooth regions.
    # --------------------------------------------------------

    print(
        "Full-resolution refinement..."
    )


    ns_result = cv2.inpaint(
        result,
        clean_mask,
        NS_RADIUS,
        cv2.INPAINT_NS,
    )


    # --------------------------------------------------------
    # Blend.
    #
    # Don't replace everything with NS result because
    # it can blur floor texture.
    # --------------------------------------------------------

    final = result.copy()


    # Use NS primarily in the interior of the mask.

    eroded_mask = cv2.erode(
        clean_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                9,
                9,
            ),
        ),
        iterations=1,
    )


    final[
        eroded_mask > 0
    ] = ns_result[
        eroded_mask > 0
    ]


    return final


# ============================================================
# LARGE REGION LAMA
# ============================================================

def run_lama(
    image,
    mask,
):

    print()
    print("========================================")
    print("LAMA REFINEMENT")
    print("========================================")


    if not USE_LAMA:

        return image


    mask_pixels = np.count_nonzero(
        mask
    )


    if mask_pixels == 0:

        print(
            "No remaining mask."
        )

        return image


    percentage = (
        100.0
        *
        mask_pixels
        /
        mask.size
    )


    print(
        f"Mask area: "
        f"{percentage:.2f}%"
    )


    print(
        "Loading LaMa..."
    )


    lama = SimpleLama()


    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


    image_pil = Image.fromarray(
        rgb
    )


    mask_pil = Image.fromarray(
        mask
    )


    print(
        "Running LaMa..."
    )


    with torch.inference_mode():

        result = lama(
            image_pil,
            mask_pil,
        )


    result_np = np.array(
        result
    )


    result_bgr = cv2.cvtColor(
        result_np,
        cv2.COLOR_RGB2BGR,
    )


    return result_bgr


# ============================================================
# EDGE / FLOOR LINE DETECTION
# ============================================================

def detect_floor_lines(
    image,
    object_mask,
):

    print()
    print("Detecting floor markings...")


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )


    h = hsv[:, :, 0]

    s = hsv[:, :, 1]

    v = hsv[:, :, 2]


    # --------------------------------------------------------
    # White markings
    # --------------------------------------------------------

    white = (
        (s < 80)
        &
        (v > 150)
    ).astype(
        np.uint8
    ) * 255


    # --------------------------------------------------------
    # Yellow boundary
    # --------------------------------------------------------

    yellow = (
        (h >= 15)
        &
        (h <= 40)
        &
        (s > 80)
        &
        (v > 80)
    ).astype(
        np.uint8
    ) * 255


    # --------------------------------------------------------
    # Blue boundary
    # --------------------------------------------------------

    blue = (
        (
            (h >= 85)
            &
            (h <= 130)
        )
        &
        (s > 70)
        &
        (v > 60)
    ).astype(
        np.uint8
    ) * 255


    lines = cv2.bitwise_or(
        white,
        yellow,
    )


    lines = cv2.bitwise_or(
        lines,
        blue,
    )


    # --------------------------------------------------------
    # Remove tiny noise.
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            7,
            7,
        ),
    )


    lines = cv2.morphologyEx(
        lines,
        cv2.MORPH_OPEN,
        kernel,
    )


    # --------------------------------------------------------
    # We only use lines where there is strong evidence.
    # --------------------------------------------------------

    return lines


# ============================================================
# RESTORE FLOOR MARKINGS
# ============================================================

def restore_floor_markings(
    original,
    reconstructed,
    object_mask,
):

    print()
    print("========================================")
    print("RESTORING FLOOR MARKINGS")
    print("========================================")


    line_mask = detect_floor_lines(
        original,
        object_mask,
    )


    result = reconstructed.copy()


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do not simply copy every white/yellow/blue pixel.
    # Only restore strong line-like structures.
    # --------------------------------------------------------

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                31,
                3,
            ),
        )
    )


    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                3,
                31,
            ),
        )
    )


    horizontal = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )


    vertical = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )


    strong_lines = cv2.bitwise_or(
        horizontal,
        vertical,
    )


    # Slightly expand line mask.

    strong_lines = cv2.dilate(
        strong_lines,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3,
            ),
        ),
    )


    # --------------------------------------------------------
    # Only restore markings inside areas that were
    # previously covered by objects.
    #
    # Visible original lines are already present.
    # --------------------------------------------------------

    restore_region = (
        (
            strong_lines > 0
        )
        &
        (
            object_mask > 0
        )
    )


    result[
        restore_region
    ] = original[
        restore_region
    ]


    return result


# ============================================================
# FINAL BLENDING
# ============================================================

def blend_result(
    original,
    reconstructed,
    mask,
):

    result = original.copy()


    result[
        mask > 0
    ] = reconstructed[
        mask > 0
    ]


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python3 background.py warehouse.jpeg"
        )

        print()

        sys.exit(1)


    image_path = (
        Path(
            sys.argv[1]
        )
        .expanduser()
        .resolve()
    )


    check_file(
        image_path
    )


    output_dir = (
        image_path.parent
    )


    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    detections_path = (
        output_dir
        / "detections.png"
    )


    mask_path = (
        output_dir
        / "object_mask.png"
    )


    reconstruction_mask_path = (
        output_dir
        / "reconstruction_mask.png"
    )


    opencv_path = (
        output_dir
        / "opencv_background.png"
    )


    final_path = (
        output_dir
        / "clean_background.png"
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("========================================")
    print("WAREHOUSE OBJECT REMOVAL V7")
    print("========================================")


    print()
    print(
        "Input:"
    )

    print(
        image_path
    )


    print()
    print(
        "Output directory:"
    )

    print(
        output_dir
    )


    # --------------------------------------------------------
    # CHECK MODELS
    # --------------------------------------------------------

    check_model_files()


    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    grounding_model = (
        load_grounding_dino()
    )


    sam_predictor = (
        load_sam2()
    )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "ERROR: Cannot read image."
        )

        sys.exit(1)


    print()
    print(
        "Image size:",
        image.shape[1],
        "x",
        image.shape[0],
    )


    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    (
        image_source,
        boxes,
        logits,
        phrases,
    ) = detect_objects(
        grounding_model,
        image_path,
    )


    # --------------------------------------------------------
    # CONVERT BOXES
    # --------------------------------------------------------

    height, width = (
        image_source.shape[:2]
    )


    detections = convert_boxes(
        boxes,
        logits,
        phrases,
        width,
        height,
    )


    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    detections = (
        remove_duplicate_boxes(
            detections
        )
    )


    print()
    print(
        "Final detections:",
        len(detections),
    )


    # --------------------------------------------------------
    # SAVE DETECTION PREVIEW
    # --------------------------------------------------------

    save_detection_preview(
        image_source,
        detections,
        detections_path,
    )


    # --------------------------------------------------------
    # SAM2
    # --------------------------------------------------------

    raw_mask = create_sam_mask(
        sam_predictor,
        image_source,
        detections,
    )


    # --------------------------------------------------------
    # REFINE MASK
    # --------------------------------------------------------

    object_mask = refine_mask(
        raw_mask
    )


    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    cv2.imwrite(
        str(mask_path),
        object_mask,
    )


    mask_percentage = (
        100.0
        *
        np.count_nonzero(
            object_mask
        )
        /
        object_mask.size
    )


    print()
    print(
        f"Object mask coverage: "
        f"{mask_percentage:.2f}%"
    )


    # --------------------------------------------------------
    # NO OBJECTS
    # --------------------------------------------------------

    if np.count_nonzero(
        object_mask
    ) == 0:

        print()
        print(
            "No objects detected."
        )


        cv2.imwrite(
            str(final_path),
            image,
        )


        sys.exit(0)


    # ========================================================
    # FLOOR CONFIDENCE
    # ========================================================

    floor_confidence = (
        create_floor_confidence(
            image,
            object_mask,
        )
    )


    # --------------------------------------------------------
    # Save for debugging.
    # --------------------------------------------------------

    floor_debug_path = (
        output_dir
        / "floor_confidence.png"
    )


    cv2.imwrite(
        str(floor_debug_path),
        floor_confidence,
    )


    # ========================================================
    # RECONSTRUCTION MASK
    # ========================================================

    # The whole object mask is reconstructed.
    #
    # But we expand it slightly so object edges
    # do not remain.

    reconstruction_mask = (
        object_mask.copy()
    )


    reconstruction_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                9,
                9,
            ),
        )
    )


    reconstruction_mask = cv2.dilate(
        reconstruction_mask,
        reconstruction_kernel,
        iterations=1,
    )


    cv2.imwrite(
        str(reconstruction_mask_path),
        reconstruction_mask,
    )


    # ========================================================
    # OPENCV RECONSTRUCTION
    # ========================================================

    opencv_background = (
        multiscale_inpaint(
            image,
            reconstruction_mask,
        )
    )


    cv2.imwrite(
        str(opencv_path),
        opencv_background,
    )


    # ========================================================
    # DETERMINE REMAINING DIFFICULT AREAS
    # ========================================================

    # --------------------------------------------------------
    # Compare the OpenCV result with original image.
    #
    # The central idea is:
    #
    # OpenCV handles large smooth floor regions.
    # LaMa handles only difficult residual areas.
    # --------------------------------------------------------

    gray_original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    gray_opencv = cv2.cvtColor(
        opencv_background,
        cv2.COLOR_BGR2GRAY,
    )


    difference = cv2.absdiff(
        gray_original,
        gray_opencv,
    )


    # Normalize.

    difference = cv2.GaussianBlur(
        difference,
        (
            9,
            9,
        ),
        0,
    )


    # --------------------------------------------------------
    # Start with object mask.
    # --------------------------------------------------------

    lama_mask = reconstruction_mask.copy()


    # --------------------------------------------------------
    # Erode slightly.
    #
    # This prevents LaMa from modifying the already
    # reconstructed boundary unnecessarily.
    # --------------------------------------------------------

    lama_mask = cv2.erode(
        lama_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                11,
                11,
            ),
        ),
        iterations=1,
    )


    # --------------------------------------------------------
    # Do NOT allow LaMa to process the whole object mask
    # blindly.
    #
    # We first use OpenCV for reconstruction.
    # --------------------------------------------------------

    remaining_area = (
        np.count_nonzero(
            lama_mask
        )
    )


    print()
    print(
        "Remaining difficult area:",
        remaining_area,
        "pixels",
    )


    # ========================================================
    # LAMA
    # ========================================================

    if remaining_area > 0:

        lama_result = run_lama(
            opencv_background,
            lama_mask,
        )

    else:

        lama_result = (
            opencv_background
        )


    # ========================================================
    # BLEND
    # ========================================================

    reconstructed = blend_result(
        image,
        lama_result,
        reconstruction_mask,
    )


    # ========================================================
    # RESTORE FLOOR MARKINGS
    # ========================================================

    final_result = (
        restore_floor_markings(
            image,
            reconstructed,
            object_mask,
        )
    )


    # ========================================================
    # FINAL COLOR SMOOTHING
    # ========================================================

    # Very small smoothing only.
    #
    # This removes hard seams without destroying
    # the floor texture.

    final_result = cv2.GaussianBlur(
        final_result,
        (
            3,
            3,
        ),
        0,
    )


    # ========================================================
    # IMPORTANT:
    # Preserve all pixels outside object regions.
    # ========================================================

    final_result[
        object_mask == 0
    ] = image[
        object_mask == 0
    ]


    # ========================================================
    # SAVE
    # ========================================================

    cv2.imwrite(
        str(final_path),
        final_result,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("========================================")
    print("PROCESS COMPLETED")
    print("========================================")


    print()
    print(
        "Input:"
    )

    print(
        image_path
    )


    print()
    print(
        "Outputs:"
    )


    print(
        "1. detections.png"
    )

    print(
        "2. object_mask.png"
    )

    print(
        "3. reconstruction_mask.png"
    )

    print(
        "4. floor_confidence.png"
    )

    print(
        "5. opencv_background.png"
    )

    print(
        "6. clean_background.png"
    )


    print()
    print(
        "Final background:"
    )

    print(
        final_path
    )


    print()
    print("========================================")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()