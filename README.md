# CSF407 Assignment-II

# AI-Based Autonomous Drone Landing Zone Detection using Semantic Segmentation, Depth Estimation and Knowledge Graph Reasoning

---

# Student Information

* Course: CSF407 – Artificial Intelligence
* Assignment: Assignment II
* Student ID: 2022A7PS0151H, 2022A4PS1508H, 2022A3PS0576H
* Submission Repository: Private GitHub Repository

---

# Project Overview

This project implements an intelligent drone landing zone detection system using:

* Deep Learning based Semantic Segmentation
* Monocular Depth Estimation
* Terrain Roughness Analysis
* Semantic Safety Analysis
* Knowledge Graph Based Reasoning
* Cost Function Optimization

The system processes aerial drone imagery and determines the safest landing location for autonomous drone deployment.

The final output includes:

* semantic segmentation,
* depth map generation,
* safe landing mask,
* optimal landing zone selection,
* semantic reasoning dashboard.

---

# Objectives

The primary objectives of the project are:

1. Perform semantic understanding of aerial scenes.
2. Estimate terrain depth from monocular images.
3. Detect safe landing regions.
4. Evaluate terrain smoothness and safety.
5. Use semantic reasoning through a knowledge graph.
6. Select the optimal landing zone using a cost function.

---

# Methodology

The complete pipeline of the project is shown below:

```text
Input Drone Image
        ↓
CNN/U-Net Semantic Segmentation
        ↓
Semantic Terrain Classification
        ↓
MiDaS Depth Estimation
        ↓
Safe Zone Mask Generation
        ↓
Terrain Roughness Computation
        ↓
Knowledge Graph Reasoning
        ↓
Cost Function Optimization
        ↓
Best Landing Zone Detection
```

---

# Technologies Used

| Component                | Technology                 |
| ------------------------ | -------------------------- |
| Programming Language     | Python                     |
| Deep Learning Framework  | PyTorch                    |
| Image Processing         | OpenCV                     |
| Numerical Computing      | NumPy                      |
| Depth Estimation         | MiDaS                      |
| Segmentation Model       | U-Net CNN                  |
| Visualization            | OpenCV                     |
| Knowledge Representation | Rule-Based Knowledge Graph |

---

# Dataset

The project uses the:

## Semantic Drone Dataset

Dataset Source:
[https://www.tugraz.at/index.php?id=22387](https://www.tugraz.at/index.php?id=22387)

Dataset contains:

* drone aerial images,
* semantic labels,
* terrain classes.

---

# Semantic Classes Used

The project uses the following semantic terrain categories:

| Terrain Class | Description          |
| ------------- | -------------------- |
| Road          | Safe landing surface |
| Grass         | Moderately safe      |
| Vegetation    | Unsafe               |
| Water         | Dangerous            |
| Building      | Unsafe               |
| Tree          | Unsafe               |
| Obstacle      | Hazardous            |

---

# Deep Learning Model

## U-Net Based CNN

A lightweight U-Net architecture was implemented for semantic segmentation.

### Encoder

* Convolution Layers
* ReLU Activations
* MaxPooling

### Decoder

* Transposed Convolutions
* Skip Connections
* Feature Reconstruction

### Output

Pixel-wise terrain classification.

---

# Depth Estimation

The project uses:

## MiDaS (Monocular Depth Estimation)

MiDaS generates depth information from a single RGB image.

Depth estimation helps evaluate:

* terrain smoothness,
* slope,
* landing feasibility.

---

# Safe Zone Detection

Safe zones are identified using:

* semantic terrain information,
* terrain roughness,
* target distance.

Safe masks are generated from:

* road regions,
* grass regions.

Morphological operations are applied for:

* noise removal,
* contour refinement.

---

# Terrain Roughness

Terrain roughness is computed using:

* Sobel Gradient Operators
* Gradient Magnitude Estimation

Lower roughness values indicate:

* flatter terrain,
* safer landing surfaces.

---

# Knowledge Graph Reasoning

A symbolic knowledge graph is used for semantic reasoning.

The knowledge graph stores:

* terrain safety,
* terrain risk,
* preferred landing surfaces,
* semantic mission rules.

Example reasoning:

| Terrain    | Safety    |
| ---------- | --------- |
| Road       | Preferred |
| Water      | Unsafe    |
| Vegetation | Risky     |
| Grass      | Moderate  |

The knowledge graph provides:

* explainable AI reasoning,
* semantic interpretability,
* mission-aware landing decisions.

---

# Cost Function

The optimal landing zone is selected using the assignment-defined cost function:

```text
Total Cost =
0.4 × Distance +
0.2 × Roughness +
0.4 × Semantic Penalty
```

Where:

* Distance → proximity to target
* Roughness → terrain smoothness
* Semantic Penalty → terrain safety

Lower total cost indicates:

* safer,
* flatter,
* more suitable landing regions.

---

# Outputs Generated

The system generates:

| File                      | Description                        |
| ------------------------- | ---------------------------------- |
| output.jpg                | Final image with landing zone      |
| output_analysis.jpg       | 6-panel analysis dashboard         |
| depth_map.jpg             | Terrain depth visualization        |
| safe_mask.jpg             | Safe landing region mask           |
| semantic_segmentation.jpg | Terrain segmentation visualization |

---

# Dashboard Description

The final dashboard contains:

1. Original Image
2. Semantic Segmentation
3. Depth Map
4. Safe Zone Mask
5. Best Landing Placement
6. Semantic Reasoning and Cost Breakdown

---

# Knowledge Graph Decision Logic

The system evaluates:

* terrain type,
* terrain smoothness,
* distance to target,
* semantic safety.

The knowledge graph then classifies the region as:

* Recommended Landing Zone
* Moderately Safe
* Unsafe Landing Zone

---

# Project Structure

```text
CSF407_2026_2022A7PS0151H_Assignment-II/
│
├── src/
│   ├── main.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── geometry.py
│   ├── knowledge_graph.py
│   ├── utils.py
│   ├── config.py
│
├── README.md
├── config.yml
├── best_model.pth
├── output.jpg
├── output_analysis.jpg
```

---

# Environment Setup

Create conda environment:

```bash
conda env create -f config.yml
```

Activate environment:

```bash
conda activate CSF407_2026_2022A7PS0151H_Assignment-II
```

---

# Running the Project

Run the main pipeline:

```bash
python src/main.py
```

Inputs required:

* image path
* target x coordinate
* target y coordinate

---

# Results

The system successfully:

* identifies safe landing zones,
* avoids unsafe terrain,
* performs semantic reasoning,
* generates explainable landing decisions,
* creates visual analysis dashboards.

The generated landing zones closely follow:

* semantic safety,
* terrain smoothness,
* assignment-defined optimization constraints.

---

# Future Improvements

Possible improvements include:

* Real-time drone deployment
* GPS integration
* Dynamic obstacle avoidance
* Reinforcement learning for landing optimization
* Multi-drone coordination
* Real-time video inference

---

# Conclusion

This project demonstrates an AI-powered autonomous drone landing system using:

* computer vision,
* deep learning,
* depth estimation,
* symbolic reasoning,
* semantic terrain understanding.

The integration of:

* CNN-based perception,
* MiDaS depth estimation,
* knowledge graph reasoning,

creates an explainable and intelligent landing zone selection framework suitable for autonomous drone applications.

