# Stereo Mosaicing & Panorama Engine

## Overview
I developed this project to implement a robust Stereo Mosaicing algorithm from scratch. 
The primary goal of this engine is to generate seamless panoramas from a continuous video sequence by leveraging
the overlapping regions between consecutive frames.

Rather than relying on high-level library functions for the core math,
I chose to implement the underlying computer vision algorithms myself—including a custom SIFT-like feature extractor
and a RANSAC motion estimator.
This approach allowed me to gain a deep, hands-on understanding of feature matching,
scale-invariant detection, and geometric transformations.

## Project Architecture
To maintain a clean, modular, and scalable codebase, I architected the project by separating the logic into distinct functional modules:

### main.py
This is the entry point of the application. I designed it to manage the end-to-end execution flow:
loading the video frames, triggering the motion calculations, initializing the panorama construction,
and finally saving the output as an aligned video sequence.
It keeps the high-level logic clean and abstracted from the heavy mathematical computations.

### features.py
In this module, I implemented the scale-invariant feature extraction.
It contains my custom logic for constructing the Difference of Gaussians (DoG) pyramid across multiple octaves and scales.
It is responsible for identifying stable local maxima (keypoints) and extracting robust,
normalized descriptors using histograms of oriented gradients. 

### motion.py
This file handles the mathematical relationship between frames. I implemented a spatial descriptor matcher
(using Euclidean distance and Lowe's ratio test) to find corresponding points.
To ensure accuracy and discard outliers, I wrote a custom implementation of the RANSAC algorithm.
This calculates the precise 2D translation vector (dx, dy) representing the camera's pure horizontal motion.

### panorama.py
This module contains the logic for physically stitching the images together.
I built functions to extract precise vertical strips from each frame based on the computed translation vectors.
To prevent harsh seams and handle auto-exposure differences,
I implemented a custom blending mechanism using feathering masks
and weighted averaging on a global expanding canvas.

### utils.py
I isolated all the file reading, writing, and formatting tasks here. 
This includes safely loading image frames from directories, extracting frames from MP4 files using OpenCV,
and compiling the final PIL image sequence back into a smooth video format.

## Technologies Used
* **Python**
* **NumPy & SciPy:** For all matrix operations, gradient calculations, and multi-dimensional filtering.
* **OpenCV:** Utilized strictly for basic frame I/O operations and color space conversions.
* **Pillow (PIL):** For final image formatting and sequence alignment.

## Installation & Execution
```bash
pip install -r requirements.txt
