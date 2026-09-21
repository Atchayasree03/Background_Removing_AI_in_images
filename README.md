

# Background_Removing_AI_in_images

## Overview

**Background_Removing_AI_in_images** is an AI-based image processing project that automatically detects unwanted objects in an image, segments them accurately, removes them, and reconstructs the missing background.

The system combines **Grounding DINO, SAM2, and LaMa** to perform the complete object-removal pipeline without requiring a predefined empty background image.

The project is designed to work with different types of images and is not limited to a particular object category or environment.

---

## Objective

The objective of this project is to automatically generate a clean version of an image by:

1. Detecting objects that need to be removed.
2. Generating accurate pixel-level segmentation masks.
3. Processing and refining the masks.
4. Removing the selected objects.
5. Reconstructing the missing regions using AI-based image inpainting.

### Input

```text
Input Image
```

### Output

```text
Clean Background Image
Object Mask
Detection Visualization
```

---

## System Architecture

```text
                 Input Image
                      │
                      ▼
            ┌───────────────────┐
            │ Object Detection  │
            │   Grounding DINO  │
            └─────────┬─────────┘
                      │
                      ▼
             Detected Objects
                      │
                      ▼
            ┌───────────────────┐
            │ Image Segmentation│
            │       SAM2        │
            └─────────┬─────────┘
                      │
                      ▼
                Object Mask
                      │
                      ▼
            ┌───────────────────┐
            │  Mask Processing  │
            │   OpenCV          │
            └─────────┬─────────┘
                      │
                      ▼
            ┌───────────────────┐
            │ AI Inpainting     │
            │       LaMa        │
            └─────────┬─────────┘
                      │
                      ▼
              Clean Image
```

---

## Technologies Used

### Grounding DINO

Grounding DINO is used for **open-vocabulary object detection**.

The system can provide text prompts describing the objects that should be detected.

For example:

```text
person . vehicle . box . furniture . container .
```

Grounding DINO returns:

* Bounding boxes
* Object labels
* Confidence scores

---

### SAM2

**Segment Anything Model 2 (SAM2)** is used to obtain a precise pixel-level segmentation of the detected objects.

Grounding DINO provides the approximate location:

```text
Bounding Box
```

SAM2 converts it into:

```text
Pixel-level Object Mask
```

This provides a more accurate region for object removal than using the bounding box directly.

---

### OpenCV

OpenCV is used for image and mask processing.

It can be used for:

* Image loading
* Image conversion
* Mask combination
* Morphological operations
* Mask dilation
* Noise removal
* Output generation

---

### LaMa

**LaMa (Large Mask Inpainting)** is used to reconstruct the regions occupied by the removed objects.

The model receives:

```text
Original Image
      +
Object Mask
      ↓
     LaMa
      ↓
Reconstructed Image
```

LaMa generates plausible image content based on the surrounding visual information.

---

## Processing Pipeline

### 1. Input Image

The system accepts an image as input.

```text
input.jpg
```

The image may contain one or multiple unwanted objects.

---

### 2. Object Detection

Grounding DINO analyzes the image using a configurable text prompt.

Example:

```text
object . person . vehicle . box . furniture .
```

The detected objects are returned with their bounding boxes and confidence scores.

---

### 3. Object Segmentation

The detected bounding boxes are passed to SAM2.

SAM2 generates a detailed segmentation mask for each detected object.

Multiple masks can then be combined into a single removal mask.

---

### 4. Mask Refinement

The generated mask is processed using image-processing operations.

Typical operations include:

* Morphological closing
* Noise removal
* Small-object filtering
* Mask dilation
* Mask smoothing

The purpose is to make sure that the complete unwanted object is covered before inpainting.

---

### 5. Object Removal and Background Reconstruction

The refined mask and original image are passed to LaMa.

LaMa reconstructs the masked regions using information from the surrounding image.

The final result is a new image where the detected objects have been removed.

---

## Output Files

The system generates the following outputs.

### `clean_background.png`

The final reconstructed image after removing the detected objects.

### `object_mask.png`

A binary mask representing the regions selected for removal.

```text
White → Object / region to remove
Black → Region to preserve
```

### `detections.png`

A visualization of the objects detected by Grounding DINO.

This is useful for verifying the detection stage before performing inpainting.

---

## Project Structure

```text
Background_Removing_AI_in_images/
│
├── background.py
├── README.md
│
├── GroundingDINO/
│   ├── groundingdino/
│   ├── weights/
│   └── ...
│
├── models/
│   └── ...
│
├── input/
│   └── input.jpg
│
└── output/
    ├── clean_background.png
    ├── object_mask.png
    └── detections.png
```

---

## Requirements

The project uses the following major libraries and models:

* Python
* PyTorch
* OpenCV
* NumPy
* Pillow
* Grounding DINO
* SAM2
* LaMa

The required model checkpoints must be downloaded and placed in their corresponding directories.

---

## Usage

Run the program by providing the input image:

```bash
python3 background.py input.jpg
```

The processing pipeline is:

```text
Input Image
     ↓
Grounding DINO
     ↓
Object Detection
     ↓
SAM2
     ↓
Object Segmentation
     ↓
Mask Refinement
     ↓
LaMa
     ↓
Clean Background Image
```

---

## No Fixed Background Image Required

One of the important characteristics of this project is that it **does not require a predefined reference background image**.

The system works directly from the input image:

```text
Input Image
     ↓
Detect Objects
     ↓
Segment Objects
     ↓
Create Mask
     ↓
Remove Objects
     ↓
Generate Background
```

This makes the approach suitable for environments where the scene or object arrangement can change.

---

## Multiple Object Removal

The system can process multiple detected objects within the same image.

```text
             Input Image
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     Object 1  Object 2  Object 3
        │         │         │
        └─────────┼─────────┘
                  ▼
            Combined Mask
                  │
                  ▼
                LaMa
                  │
                  ▼
        Clean Background Image
```

---

## Advantages

* No manually created background image is required.
* Supports multiple objects.
* Uses open-vocabulary object detection.
* Produces pixel-level object masks.
* Uses AI-based image inpainting.
* Can be adapted to different environments.
* Object categories can be changed using text prompts.
* The detection, segmentation, and inpainting stages are modular.

---

## Limitations

The reconstructed background is an **AI-generated reconstruction**. It is not guaranteed to contain the exact pixels that originally existed behind the removed object.

The quality depends on:

* Object detection accuracy
* Segmentation accuracy
* Mask quality
* Object size
* Image resolution
* Amount of visible surrounding background
* Complexity of the scene

If an object completely hides a large region and there is no information about what exists behind it, the inpainting model has to generate a plausible reconstruction.

---

## Future Improvements

Possible improvements include:

* Better object detection
* Custom object detection models
* Improved segmentation
* Automatic mask refinement
* GPU acceleration
* Faster inference
* Batch image processing
* Web-based interface
* REST API integration
* User-selectable objects for removal
* Confidence-based object filtering

---

## Core Concept

The project follows a modular AI pipeline:

```text
             IMAGE
               │
               ▼
        Object Detection
        Grounding DINO
               │
               ▼
         Object Location
               │
               ▼
        Object Segmentation
              SAM2
               │
               ▼
          Object Mask
               │
               ▼
         Mask Refinement
             OpenCV
               │
               ▼
          AI Inpainting
              LaMa
               │
               ▼
       Clean Background
```

### In short

**Detect → Segment → Mask → Remove → Reconstruct**

The project focuses specifically on **removing unwanted objects from individual images and generating a clean, reconstructed background image**.
