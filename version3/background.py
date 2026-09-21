#!/usr/bin/env python3
"""
============================================================
WAREHOUSE OBJECT REMOVAL - CPU ONLY - VERSION 6
============================================================

Pipeline:

    Input Image
         |
         v
    SAM2 initialization
         |
         v
    GroundingDINO
         |
         v
    Object Bounding Boxes
         |
         v
    Duplicate / Bad Box Filtering
         |
         v
       SAM2
         |
         v
    Box Guided Segmentation
         |
         v
    SAM Candidate Selection
         |
         v
    Mask Refinement
         |
         v
    Combined Object Mask
         |
         v
        LaMa
         |
         v
    Clean Background

Usage:

    cd ~/Pictures/background_predicting

    python3 background.py warehouse.jpeg

Expected files:

    ~/Pictures/background_predicting/
        background.py
        warehouse.jpeg
        GroundingDINO/
            groundingdino/
            weights/
                groundingdino_swint_ogc.pth

    ~/sam2/
        sam2/
            __init__.py
            build_sam.py
            configs/
                sam2.1/
                    sam2.1_hiera_s.yaml
        checkpoints/
            sam2.1_hiera_small.pt

Outputs:

    clean_background.png
    object_mask.png
    detections.png
    sam2_preview.png

IMPORTANT:
    This script is CPU ONLY.
    CUDA is disabled before torch is imported.
"""

# ============================================================
# CPU ENVIRONMENT
# ============================================================

import os
import sys
from pathlib import Path

# Force CPU before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# GroundingDINO repository:
# ~/Pictures/background_predicting/GroundingDINO
GROUNDINGDINO_DIR = BASE_DIR / "GroundingDINO"

# SAM2 repository:
# ~/sam2
SAM2_DIR = Path.home() / "sam2"

# IMPORTANT:
# Python must see the repository PARENT.
#
# Correct:
#     /home/atchayasree/sam2
#
# Not:
#     /home/atchayasree/sam2/sam2
sys.path.insert(0, str(SAM2_DIR))
sys.path.insert(0, str(GROUNDINGDINO_DIR))

# Make SAM2 repository available for Hydra/package discovery.
old_pythonpath = os.environ.get("PYTHONPATH", "")

if old_pythonpath:
    if str(SAM2_DIR) not in old_pythonpath.split(":"):
        os.environ["PYTHONPATH"] = (
            f"{SAM2_DIR}:{old_pythonpath}"
        )
else:
    os.environ["PYTHONPATH"] = str(SAM2_DIR)

# ============================================================
# THIRD-PARTY IMPORTS
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

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# -------------------------------
# GroundingDINO
# -------------------------------

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

# -------------------------------
# SAM2
# -------------------------------

# IMPORTANT:
# SAM2's build_sam2() expects a Hydra
# config name, not the absolute filesystem path.
#
# The SAM2 repository contains:
#
# ~/sam2/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
#
# SAM2's build_sam.py is designed to use:
#
# configs/sam2.1/sam2.1_hiera_s.yaml
SAM2_CONFIG = "sam2.1/sam2.1_hiera_s.yaml"

SAM2_CONFIG_FILE = (
    SAM2_DIR
    / "sam2"
    / "configs"
    / "sam2.1"
    / "sam2.1_hiera_s.yaml"
)

SAM2_CHECKPOINT = (
    SAM2_DIR
    / "checkpoints"
    / "sam2.1_hiera_small.pt"
)

# -------------------------------
# Device
# -------------------------------

DEVICE = "cpu"

# ============================================================
# GROUNDINGDINO SETTINGS
# ============================================================

# IMPORTANT:
# The target is NOT just boxes. We want the final image to contain
# the warehouse FLOOR only. Therefore pallets are also foreground
# and must be removed.
#
# We run GroundingDINO in two semantic groups. This prevents a
# pallet detection from being suppressed just because it overlaps
# a box/container sitting on top of it.
PALLET_PROMPT = (
    "pallet . "
    "wooden pallet . "
    "stack of pallets . "
)

ITEM_PROMPT = (
    "box . "
    "container . "
    "crate . "
)

# Pallets in this camera view are relatively difficult for
# GroundingDINO, so use a lower threshold for the pallet pass.
PALLET_BOX_THRESHOLD = 0.12
PALLET_TEXT_THRESHOLD = 0.08

ITEM_BOX_THRESHOLD = 0.15
ITEM_TEXT_THRESHOLD = 0.10

# Ignore very small detections.
MIN_OBJECT_AREA = 800

# Keep pallet and item detections independently. This is important:
# a large number of box/container detections must NEVER consume the
# slots needed by pallet detections.
MAX_PALLETS = 20
MAX_ITEMS = 35
MAX_OBJECTS = MAX_PALLETS + MAX_ITEMS

# Duplicate filtering is performed ONLY within the same semantic
# group. A pallet and a box may overlap heavily and BOTH must stay.
DUPLICATE_IOU_THRESHOLD = 0.55
DUPLICATE_CONTAINMENT_THRESHOLD = 0.78


# ============================================================
# MASK SETTINGS
# ============================================================

# Slightly larger than the previous version because we want to
# remove visible pallet edges/slats as well as the containers.
MASK_DILATION = 7
MORPH_KERNEL_SIZE = 5

# Extra SAM2 padding around detected pallets.
PALLET_MASK_PAD = 12
ITEM_MASK_PAD = 7

# ============================================================
# SAM2 CANDIDATE SETTINGS
# ============================================================

MIN_MASK_BOX_RATIO = 0.05
MAX_MASK_BOX_RATIO = 1.05

# ============================================================
# DEVICE INFORMATION
# ============================================================

print()
print("========================================")
print("CPU MODE")
print("========================================")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", DEVICE)
print("========================================")


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path):
    """Stop immediately if a required file is missing."""

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
# MODEL FILE CHECK
# ============================================================

def check_model_files():

    print()
    print("========================================")
    print("CHECKING MODEL FILES")
    print("========================================")

    print()
    print("GroundingDINO config:")
    print(GROUNDING_CONFIG)
    check_file(GROUNDING_CONFIG)

    print()
    print("GroundingDINO checkpoint:")
    print(GROUNDING_CHECKPOINT)
    check_file(GROUNDING_CHECKPOINT)

    print()
    print("SAM2 config name:")
    print(SAM2_CONFIG)

    print()
    print("SAM2 config file:")
    print(SAM2_CONFIG_FILE)
    check_file(SAM2_CONFIG_FILE)

    print()
    print("SAM2 checkpoint:")
    print(SAM2_CHECKPOINT)
    check_file(SAM2_CHECKPOINT)

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

    print()
    print("GroundingDINO loaded successfully.")

    return model


# ============================================================
# LOAD SAM2
# ============================================================

def load_sam2():

    print()
    print("========================================")
    print("LOADING SAM2")
    print("========================================")

    print()
    print("SAM2 repository:")
    print(SAM2_DIR)

    print()
    print("SAM2 config name:")
    print(SAM2_CONFIG)

    print()
    print("SAM2 config file:")
    print(SAM2_CONFIG_FILE)

    print()
    print("SAM2 checkpoint:")
    print(SAM2_CHECKPOINT)

    # --------------------------------------------------------
    # Verify SAM2 package.
    # --------------------------------------------------------

    import sam2

    sam2_file = getattr(
        sam2,
        "__file__",
        None,
    )

    print()
    print("SAM2 Python package:")
    print(sam2_file)

    expected_init = (
        SAM2_DIR
        / "sam2"
        / "__init__.py"
    )

    if not expected_init.exists():
        raise FileNotFoundError(
            "SAM2 package __init__.py not found:\n"
            f"{expected_init}"
        )

    # --------------------------------------------------------
    # Verify SAM2 config directory/file.
    # --------------------------------------------------------

    sam2_config_dir = (
        SAM2_DIR
        / "sam2"
        / "configs"
    )

    if not sam2_config_dir.exists():
        raise FileNotFoundError(
            "SAM2 config directory not found:\n"
            f"{sam2_config_dir}"
        )

    if not SAM2_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "SAM2 config file not found:\n"
            f"{SAM2_CONFIG_FILE}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # build_sam2() internally calls Hydra compose().
    # Therefore Hydra MUST be initialized before build_sam2().
    #
    # The problem in the previous versions was:
    #
    #   GroundingDINO initialized Hydra
    #          ->
    #   SAM2 tried to initialize Hydra again
    #
    # or:
    #
    #   We cleared Hydra
    #          ->
    #   SAM2 called compose()
    #          ->
    #   Hydra was not initialized
    #
    # This function solves it by explicitly initializing Hydra
    # with SAM2's config directory and keeping build_sam2()
    # inside that Hydra context.
    # --------------------------------------------------------

    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_dir

    # Remove any Hydra state left by another library.
    if GlobalHydra.instance().is_initialized():
        print()
        print("Clearing existing Hydra state...")
        GlobalHydra.instance().clear()

    print()
    print("Initializing Hydra for SAM2...")
    print("Config directory:")
    print(sam2_config_dir)

    # Import only after the repository path has been installed.
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Hydra requires an absolute config directory.
    with initialize_config_dir(
        version_base=None,
        config_dir=str(
            sam2_config_dir.resolve()
        ),
    ):

        print()
        print("Building SAM2 model...")

        model = build_sam2(
            SAM2_CONFIG,
            str(SAM2_CHECKPOINT),
            device=DEVICE,
        )

    # The Hydra context has now finished.
    # build_sam2 has already constructed the model.
    predictor = SAM2ImagePredictor(model)

    print()
    print("SAM2 loaded successfully.")

    return predictor


# ============================================================
# GROUNDINGDINO DETECTION
# ============================================================

def detect_objects(
    grounding_model,
    image_path,
):

    print()
    print("========================================")
    print("GROUNDINGDINO OBJECT DETECTION")
    print("========================================")

    image_source, image = load_image(
        str(image_path)
    )

    all_boxes = []
    all_logits = []
    all_phrases = []
    all_types = []

    # --------------------------------------------------------
    # PASS 1: PALLETS
    # --------------------------------------------------------

    print()
    print("PALLET DETECTION PROMPT:")
    print(PALLET_PROMPT)

    pallet_boxes, pallet_logits, pallet_phrases = predict(
        model=grounding_model,
        image=image,
        caption=PALLET_PROMPT,
        box_threshold=PALLET_BOX_THRESHOLD,
        text_threshold=PALLET_TEXT_THRESHOLD,
        device=DEVICE,
    )

    print()
    print("Raw pallet detections:", len(pallet_boxes))

    for i, phrase in enumerate(pallet_phrases):
        score = float(pallet_logits[i])
        print(f"  PALLET {i + 1}: {phrase} confidence={score:.3f}")

    for i in range(len(pallet_boxes)):
        all_boxes.append(pallet_boxes[i])
        all_logits.append(pallet_logits[i])
        all_phrases.append(str(pallet_phrases[i]))
        all_types.append("pallet")

    # --------------------------------------------------------
    # PASS 2: BOX / CONTAINER / CRATE
    # --------------------------------------------------------

    print()
    print("ITEM DETECTION PROMPT:")
    print(ITEM_PROMPT)

    item_boxes, item_logits, item_phrases = predict(
        model=grounding_model,
        image=image,
        caption=ITEM_PROMPT,
        box_threshold=ITEM_BOX_THRESHOLD,
        text_threshold=ITEM_TEXT_THRESHOLD,
        device=DEVICE,
    )

    print()
    print("Raw item detections:", len(item_boxes))

    for i, phrase in enumerate(item_phrases):
        score = float(item_logits[i])
        print(f"  ITEM {i + 1}: {phrase} confidence={score:.3f}")

    for i in range(len(item_boxes)):
        all_boxes.append(item_boxes[i])
        all_logits.append(item_logits[i])
        all_phrases.append(str(item_phrases[i]))
        all_types.append("item")

    print()
    print("Total raw detections:", len(all_boxes))

    return (
        image_source,
        image,
        all_boxes,
        all_logits,
        all_phrases,
        all_types,
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b,
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# BOX OVERLAP HELPERS
# ============================================================

def box_area(box):

    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_containment(box_a, box_b):
    """
    Intersection divided by the area of the smaller box.

    This is useful for GroundingDINO duplicates where one prompt
    produces a slightly larger box around the same physical object.
    """

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    smaller = min(
        box_area(box_a),
        box_area(box_b),
    )

    if smaller <= 0:
        return 0.0

    return intersection / smaller


def is_duplicate_detection(candidate, existing):
    """
    Decide whether two GroundingDINO detections are likely to be
    the same physical object.

    We remove detections when either:
      1. IoU is high, or
      2. one box almost completely contains the other.

    This is deliberately less aggressive than merging all overlapping
    boxes, because stacked warehouse objects can legitimately overlap.
    """

    box_a = candidate["box"]
    box_b = existing["box"]

    # NEVER suppress a pallet because it overlaps a box/container.
    # They are separate foreground classes for our ground-only goal.
    if candidate.get("object_type") != existing.get("object_type"):
        return False, calculate_iou(box_a, box_b), calculate_containment(box_a, box_b)

    iou = calculate_iou(box_a, box_b)
    containment = calculate_containment(box_a, box_b)

    if iou >= DUPLICATE_IOU_THRESHOLD:
        return True, iou, containment

    if containment >= DUPLICATE_CONTAINMENT_THRESHOLD:
        return True, iou, containment

    return False, iou, containment


# ============================================================
# BOX CONVERSION AND FILTERING
# ============================================================

def _to_cpu_tensor(data, name="data"):
    """
    Convert GroundingDINO output to a CPU float tensor.

    Depending on the installed GroundingDINO / PyTorch combination,
    predict() may return:
        - torch.Tensor
        - numpy.ndarray
        - list/tuple of tensors
        - list/tuple of lists

    The previous code assumed a tensor and called .detach() directly,
    which caused:
        AttributeError: 'list' object has no attribute 'detach'
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().float()

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return torch.empty((0,), dtype=torch.float32)

        converted = []
        for item in data:
            if isinstance(item, torch.Tensor):
                converted.append(item.detach().cpu().float())
            elif isinstance(item, np.ndarray):
                converted.append(torch.from_numpy(item).float())
            else:
                converted.append(torch.as_tensor(item, dtype=torch.float32))

        try:
            return torch.stack(converted)
        except RuntimeError:
            return torch.as_tensor(data, dtype=torch.float32)

    return torch.as_tensor(data, dtype=torch.float32)


def convert_boxes(
    boxes,
    logits,
    phrases,
    object_types,
    image_width,
    image_height,
):
    """
    Convert normalized GroundingDINO boxes to pixel XYXY boxes.

    Handles GroundingDINO returning boxes as either a Tensor or a list.

    Pallets and items are filtered independently so that box/container
    detections cannot push pallet detections out of MAX_OBJECTS.
    """

    print()
    print("========================================")
    print("CONVERTING DETECTION BOXES")
    print("========================================")

    if boxes is None or len(boxes) == 0:
        return []

    # --------------------------------------------------------
    # Normalize GroundingDINO outputs.
    # --------------------------------------------------------

    boxes_tensor = _to_cpu_tensor(boxes, "boxes")
    logits_tensor = _to_cpu_tensor(logits, "logits")

    print("GroundingDINO boxes type:", type(boxes).__name__)
    print("Normalized boxes shape:", tuple(boxes_tensor.shape))

    # Make boxes shape [N, 4].
    if boxes_tensor.ndim == 1:
        if boxes_tensor.numel() == 4:
            boxes_tensor = boxes_tensor.reshape(1, 4)
        else:
            raise ValueError(
                f"Unexpected GroundingDINO box shape: "
                f"{tuple(boxes_tensor.shape)}"
            )

    if boxes_tensor.ndim != 2 or boxes_tensor.shape[1] != 4:
        raise ValueError(
            "GroundingDINO boxes must have shape [N, 4], got "
            f"{tuple(boxes_tensor.shape)}"
        )

    # Flatten logits to [N].
    logits_tensor = logits_tensor.reshape(-1)

    count = min(
        boxes_tensor.shape[0],
        logits_tensor.shape[0],
        len(phrases),
        len(object_types),
    )

    if count == 0:
        return []

    boxes_tensor = boxes_tensor[:count]
    logits_tensor = logits_tensor[:count]

    # Normalized cx, cy, w, h -> pixels.
    scale = torch.tensor(
        [
            image_width,
            image_height,
            image_width,
            image_height,
        ],
        dtype=torch.float32,
    )

    boxes_scaled = boxes_tensor * scale

    detections = []

    for index in range(count):

        cx, cy, w, h = boxes_scaled[index].tolist()

        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = int(round(cx + w / 2))
        y2 = int(round(cy + h / 2))

        # Clamp.
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))

        width = x2 - x1
        height = y2 - y1
        area = width * height

        if (
            width <= 0
            or height <= 0
            or area < MIN_OBJECT_AREA
        ):
            continue

        score = float(logits_tensor[index].item())

        phrase = str(phrases[index]).strip()

        object_type = str(
            object_types[index]
        ).strip().lower()

        if object_type not in ("pallet", "item"):
            object_type = "item"

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "score": score,
                "phrase": phrase,
                "object_type": object_type,
                "original_index": index,
            }
        )

    print()
    print("Valid boxes before duplicate filtering:", len(detections))

    # --------------------------------------------------------
    # IMPORTANT:
    # Filter pallets and items separately.
    # --------------------------------------------------------

    pallet_candidates = [
        d for d in detections
        if d["object_type"] == "pallet"
    ]

    item_candidates = [
        d for d in detections
        if d["object_type"] == "item"
    ]

    def filter_group(group, group_name, maximum):
        group = sorted(
            group,
            key=lambda d: d["score"],
            reverse=True,
        )

        filtered_group = []
        duplicate_count = 0

        for candidate in group:

            duplicate = False

            for existing in filtered_group:

                is_dup, iou, containment = (
                    is_duplicate_detection(
                        candidate,
                        existing,
                    )
                )

                if is_dup:
                    duplicate = True
                    duplicate_count += 1

                    print(
                        f"  [{group_name}] duplicate removed: "
                        f"{candidate['phrase']} "
                        f"{candidate['score']:.3f} | "
                        f"IoU={iou:.2f} "
                        f"containment={containment:.2f} | "
                        f"kept {existing['phrase']} "
                        f"{existing['score']:.3f}"
                    )
                    break

            if not duplicate:
                filtered_group.append(candidate)

            if len(filtered_group) >= maximum:
                break

        print()
        print(
            f"{group_name} candidates: {len(group)}"
        )
        print(
            f"{group_name} duplicates removed: "
            f"{duplicate_count}"
        )
        print(
            f"{group_name} kept: "
            f"{len(filtered_group)}"
        )

        return filtered_group

    print()
    print("----------------------------------------")
    print("PALLET FILTERING")
    print("----------------------------------------")

    filtered_pallets = filter_group(
        pallet_candidates,
        "PALLET",
        MAX_PALLETS,
    )

    print()
    print("----------------------------------------")
    print("ITEM FILTERING")
    print("----------------------------------------")

    filtered_items = filter_group(
        item_candidates,
        "ITEM",
        MAX_ITEMS,
    )

    # --------------------------------------------------------
    # Put pallets first.
    #
    # This guarantees that, if a future safety limit is reached,
    # pallets are preserved before item detections.
    # --------------------------------------------------------

    filtered = (
        filtered_pallets
        + filtered_items
    )

    filtered = filtered[:MAX_OBJECTS]

    print()
    print("========================================")
    print("FINAL DETECTIONS")
    print("========================================")
    print("Pallets kept:", len(filtered_pallets))
    print("Items kept:", len(filtered_items))
    print("Total kept:", len(filtered))

    for i, detection in enumerate(filtered):

        print(
            f"  {i + 1}: "
            f"{detection['object_type']} / "
            f"{detection['phrase']} "
            f"{detection['score']:.3f} "
            f"box={detection['box']}"
        )

    return filtered


# ============================================================
# MASK QUALITY SCORE
# ============================================================

def calculate_mask_quality(
    mask,
    box,
    sam_score,
):

    x1, y1, x2, y2 = box

    box_width = max(
        1,
        x2 - x1,
    )

    box_height = max(
        1,
        y2 - y1,
    )

    box_area = (
        box_width
        * box_height
    )

    mask_area = float(
        np.count_nonzero(mask)
    )

    if mask_area <= 0:
        return -999.0

    mask_box_ratio = (
        mask_area
        / box_area
    )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        return -999.0

    mask_x1 = int(xs.min())
    mask_y1 = int(ys.min())
    mask_x2 = int(xs.max())
    mask_y2 = int(ys.max())

    mask_bbox_area = (
        max(
            1,
            mask_x2 - mask_x1,
        )
        *
        max(
            1,
            mask_y2 - mask_y1,
        )
    )

    # Start with SAM's own confidence.
    quality = (
        float(sam_score)
        * 2.0
    )

    # Penalize masks that fill almost the entire box.
    if mask_box_ratio > 0.90:

        quality -= (
            mask_box_ratio - 0.90
        ) * 2.0

    # Very tiny masks are suspicious.
    if mask_box_ratio < MIN_MASK_BOX_RATIO:
        quality -= 2.0

    # Prevent unused variable warnings.
    _ = mask_bbox_area

    if mask_box_ratio > MAX_MASK_BOX_RATIO:
        quality -= 5.0

    return quality


# ============================================================
# SAM2 OBJECT SEGMENTATION
# ============================================================

def create_object_mask(
    predictor,
    image_rgb,
    detections,
):

    print()
    print("========================================")
    print("SAM2 OBJECT SEGMENTATION")
    print("========================================")

    height, width = image_rgb.shape[:2]

    final_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    if len(detections) == 0:

        print()
        print("No detections available.")

        return final_mask

    # --------------------------------------------------------
    # Encode image once.
    # --------------------------------------------------------

    print()
    print("Encoding image with SAM2...")

    with torch.inference_mode():
        predictor.set_image(
            image_rgb
        )

    print(
        "Image encoded successfully."
    )

    # --------------------------------------------------------
    # Process detections.
    # --------------------------------------------------------

    for index, detection in enumerate(
        detections
    ):

        box = detection["box"]

        print()
        print(
            "----------------------------------------"
        )

        print(
            f"Object {index + 1}/"
            f"{len(detections)}"
        )

        print(
            "Type:",
            detection["phrase"],
        )

        print(
            "Confidence:",
            f"{detection['score']:.3f}",
        )

        print(
            "Box:",
            box,
        )

        box_np = np.asarray(
            box,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Generate candidate masks.
        # ----------------------------------------------------

        try:

            with torch.inference_mode():

                masks, scores, _ = (
                    predictor.predict(
                        box=box_np,
                        multimask_output=True,
                    )
                )

        except Exception as exc:

            print()
            print("SAM2 error:")
            print(exc)

            continue

        if (
            masks is None
            or len(masks) == 0
        ):

            print(
                "No mask generated."
            )

            continue

        # ----------------------------------------------------
        # Evaluate candidates.
        # ----------------------------------------------------

        candidates = []

        for candidate_index in range(
            len(masks)
        ):

            candidate_mask = (
                masks[candidate_index]
            )

            candidate_mask = (
                candidate_mask
                .astype(np.uint8)
            )

            sam_score = float(
                scores[candidate_index]
            )

            quality = (
                calculate_mask_quality(
                    candidate_mask,
                    box,
                    sam_score,
                )
            )

            candidates.append(
                (
                    quality,
                    candidate_index,
                    sam_score,
                    candidate_mask,
                )
            )

            print(
                f"  Candidate "
                f"{candidate_index + 1}: "
                f"SAM={sam_score:.3f} "
                f"quality={quality:.3f} "
                f"pixels="
                f"{np.count_nonzero(candidate_mask)}"
            )

        if not candidates:
            continue

        # Pallet masks often have a high-quality candidate that captures
        # only the top board. For pallets, favor a candidate with larger
        # coverage of the detected box while still respecting SAM score.
        if detection.get("object_type") == "pallet":
            def pallet_rank(candidate):
                quality, _, sam_score, candidate_mask = candidate
                mask_area = float(np.count_nonzero(candidate_mask))
                bx1, by1, bx2, by2 = box
                bbox_area = max(1, (bx2 - bx1) * (by2 - by1))
                coverage_ratio = min(1.5, mask_area / bbox_area)
                return quality + (0.65 * coverage_ratio) + (0.35 * sam_score)

            candidates.sort(key=pallet_rank, reverse=True)
        else:
            candidates.sort(
                key=lambda x: x[0],
                reverse=True,
            )

        (
            best_quality,
            best_index,
            best_sam_score,
            best_mask,
        ) = candidates[0]

        print()
        print(
            "Selected candidate:",
            best_index + 1,
        )

        print(
            "Selected SAM score:",
            f"{best_sam_score:.3f}",
        )

        print(
            "Selected quality:",
            f"{best_quality:.3f}",
        )

        # ----------------------------------------------------
        # Restrict mask to a small region around the box.
        # This prevents an accidental SAM mask from consuming
        # unrelated warehouse background.
        # ----------------------------------------------------

        x1, y1, x2, y2 = box

        if detection.get("object_type") == "pallet":
            pad = PALLET_MASK_PAD
        else:
            pad = ITEM_MASK_PAD

        rx1 = max(
            0,
            x1 - pad,
        )

        ry1 = max(
            0,
            y1 - pad,
        )

        rx2 = min(
            width,
            x2 + pad,
        )

        ry2 = min(
            height,
            y2 + pad,
        )

        restricted_mask = np.zeros_like(
            best_mask,
            dtype=np.uint8,
        )

        restricted_mask[
            ry1:ry2,
            rx1:rx2
        ] = best_mask[
            ry1:ry2,
            rx1:rx2
        ]

        # ----------------------------------------------------
        # Combine.
        # ----------------------------------------------------

        final_mask[
            restricted_mask > 0
        ] = 255

    return final_mask


# ============================================================
# MASK REFINEMENT
# ============================================================

def refine_mask(mask):

    print()
    print("========================================")
    print("REFINING FINAL MASK")
    print("========================================")

    if mask is None:
        return None

    if np.count_nonzero(mask) == 0:
        print(
            "Mask is empty."
        )
        return mask.astype(
            np.uint8
        )

    # Binary mask.
    mask = np.where(
        mask > 0,
        255,
        0,
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Close small gaps.
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE,
        ),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    # --------------------------------------------------------
    # Remove tiny isolated components.
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(
        mask
    )

    for label in range(
        1,
        num_labels,
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA,
        ]

        if area >= 300:

            cleaned[
                labels == label
            ] = 255

    mask = cleaned

    # --------------------------------------------------------
    # Slight dilation.
    # --------------------------------------------------------

    if MASK_DILATION > 0:

        dilation_size = (
            MASK_DILATION * 2
        ) + 1

        dilation_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    dilation_size,
                    dilation_size,
                ),
            )
        )

        mask = cv2.dilate(
            mask,
            dilation_kernel,
            iterations=1,
        )

    print(
        "Final mask pixels:",
        int(np.count_nonzero(mask)),
    )

    return mask


# ============================================================
# SAVE MASK
# ============================================================

def save_mask(
    mask,
    output_path,
):

    cv2.imwrite(
        str(output_path),
        mask,
    )

    print()
    print(
        "Object mask saved:"
    )

    print(output_path)


# ============================================================
# DETECTION PREVIEW
# ============================================================

def save_detection_preview(
    image_source,
    detections,
    output_path,
):

    preview = image_source.copy()

    for index, detection in enumerate(
        detections
    ):

        x1, y1, x2, y2 = (
            detection["box"]
        )

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"{detection['object_type']} "
            f"{detection['score']:.2f}"
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
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        preview,
    )

    print()
    print(
        "Detection preview saved:"
    )

    print(output_path)


# ============================================================
# SAM2 PREVIEW
# ============================================================

def save_sam2_preview(
    image_source,
    mask,
    output_path,
):

    preview = image_source.copy()

    mask_bool = (
        mask > 0
    )

    # BGR image.
    overlay = preview.copy()

    # Red overlay.
    overlay[
        mask_bool
    ] = (
        0,
        0,
        255,
    )

    if np.any(mask_bool):

        preview[mask_bool] = (
            0.45
            * preview[mask_bool]
            + 0.55
            * overlay[mask_bool]
        ).astype(
            np.uint8
        )

    cv2.imwrite(
        str(output_path),
        preview,
    )

    print()
    print(
        "SAM2 preview saved:"
    )

    print(output_path)


# ============================================================
# LAMA
# ============================================================

def run_lama(
    image_source,
    mask,
    output_path,
):

    print()
    print("========================================")
    print("LOADING LAMA")
    print("========================================")

    from simple_lama_inpainting import (
        SimpleLama,
    )

    lama = SimpleLama()

    print()
    print(
        "LaMa loaded."
    )

    print()
    print(
        "Running LaMa..."
    )

    print()
    print(
        "CPU inpainting can take some time."
    )

    # --------------------------------------------------------
    # Convert BGR -> RGB.
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_source,
        cv2.COLOR_BGR2RGB,
    )

    image_pil = Image.fromarray(
        image_rgb
    ).convert("RGB")

    # --------------------------------------------------------
    # LaMa:
    #
    # WHITE = remove
    # BLACK = preserve
    # --------------------------------------------------------

    mask_pil = Image.fromarray(
        mask
    ).convert("L")

    try:

        with torch.inference_mode():

            result = lama(
                image_pil,
                mask_pil,
            )

    except Exception as exc:

        print()
        print("LaMa ERROR:")
        print(exc)

        raise

    if isinstance(
        result,
        Image.Image,
    ):

        result = result.convert(
            "RGB"
        )

    else:

        result = Image.fromarray(
            np.asarray(result)
            .astype(np.uint8)
        ).convert(
            "RGB"
        )

    result.save(
        str(output_path)
    )

    print()
    print(
        "LaMa completed."
    )

    print()
    print(
        "Clean background saved:"
    )

    print(output_path)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # INPUT
    # ========================================================

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
        Path(sys.argv[1])
        .expanduser()
        .resolve()
    )

    check_file(
        image_path
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    output_dir = (
        image_path.parent
    )

    clean_background_path = (
        output_dir
        / "clean_background.png"
    )

    object_mask_path = (
        output_dir
        / "object_mask.png"
    )

    detections_path = (
        output_dir
        / "detections.png"
    )

    sam2_preview_path = (
        output_dir
        / "sam2_preview.png"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("========================================")
    print("WAREHOUSE OBJECT REMOVAL - VERSION 6")
    print("========================================")

    print()
    print("Input:")
    print(image_path)

    print()
    print("Output directory:")
    print(output_dir)

    # ========================================================
    # MODEL CHECK
    # ========================================================

    check_model_files()

    # ========================================================
    # LOAD SAM2 FIRST
    #
    # SAM2 uses Hydra. Load it before GroundingDINO so its
    # Hydra configuration is initialized in a clean context.
    # ========================================================

    sam_predictor = (
        load_sam2()
    )

    # ========================================================
    # LOAD GROUNDINGDINO
    #
    # GroundingDINO can now initialize/use its own Hydra state
    # after SAM2 has finished construction.
    # ========================================================

    grounding_model = (
        load_grounding_dino()
    )

    # ========================================================
    # DETECTION
    # ========================================================

    (
        image_source,
        image_tensor,
        raw_boxes,
        raw_logits,
        raw_phrases,
        raw_types,
    ) = detect_objects(
        grounding_model,
        image_path,
    )

    # Prevent unused-variable warnings.
    _ = image_tensor

    # ========================================================
    # IMAGE SIZE
    # ========================================================

    image_height, image_width = (
        image_source.shape[:2]
    )

    print()
    print(
        "Image size:",
        image_width,
        "x",
        image_height,
    )

    # ========================================================
    # CONVERT + FILTER
    # ========================================================

    detections = convert_boxes(
        raw_boxes,
        raw_logits,
        raw_phrases,
        raw_types,
        image_width,
        image_height,
    )

    # ========================================================
    # DETECTION PREVIEW
    # ========================================================

    save_detection_preview(
        image_source,
        detections,
        detections_path,
    )

    # ========================================================
    # RGB IMAGE
    # ========================================================

    image_rgb = cv2.cvtColor(
        image_source,
        cv2.COLOR_BGR2RGB,
    )

    # ========================================================
    # SAM2
    # ========================================================

    mask = create_object_mask(
        sam_predictor,
        image_rgb,
        detections,
    )

    # ========================================================
    # MASK REFINEMENT
    # ========================================================

    mask = refine_mask(
        mask
    )

    # ========================================================
    # MASK STATISTICS
    # ========================================================

    mask_pixels = int(
        np.count_nonzero(mask)
    )

    total_pixels = (
        mask.shape[0]
        * mask.shape[1]
    )

    coverage = (
        100.0
        * mask_pixels
        / total_pixels
        if total_pixels > 0
        else 0.0
    )

    print()
    print("========================================")
    print("MASK STATISTICS")
    print("========================================")

    print()
    print(
        "Mask pixels:",
        mask_pixels,
    )

    print(
        f"Mask coverage: "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # SAVE SAM2 PREVIEW
    # ========================================================

    save_sam2_preview(
        image_source,
        mask,
        sam2_preview_path,
    )

    # ========================================================
    # SAVE MASK
    # ========================================================

    save_mask(
        mask,
        object_mask_path,
    )

    # ========================================================
    # LAMA
    # ========================================================

    if mask_pixels > 0:

        run_lama(
            image_source,
            mask,
            clean_background_path,
        )

    else:

        print()
        print(
            "WARNING:"
        )

        print(
            "No object mask was generated."
        )

        print(
            "LaMa will NOT be executed."
        )

        # Keep an output file so the pipeline
        # has a deterministic output.
        cv2.imwrite(
            str(clean_background_path),
            image_source,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("========================================")
    print("PROCESS COMPLETED")
    print("========================================")

    print()
    print("Input:")
    print(image_path)

    print()
    print("Outputs:")

    print(
        "1. Clean background:"
    )
    print(
        clean_background_path
    )

    print(
        "2. Object mask:"
    )
    print(
        object_mask_path
    )

    print(
        "3. GroundingDINO detections:"
    )
    print(
        detections_path
    )

    print(
        "4. SAM2 preview:"
    )
    print(
        sam2_preview_path
    )

    print()
    print("========================================")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
