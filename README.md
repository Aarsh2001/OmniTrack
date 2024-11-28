# OmniTrack

Thesis document can be found [here](https://drive.google.com/file/d/1zU0AkN80Zk_GOe6Yb8sdnuzQ_Ag4ZJWa/view?usp=sharing)

# Master's Thesis Codebase

## Overview

This repository contains all the code developed and used for my master's thesis, focusing on mitigating the projection re-orientation problem in the 360VOT Test Set. The work includes benchmarking, qualitative and quantitative analysis, and the development of new methods to improve object tracking in 360-degree video.

## Directory Structure

### 1. **360VOT/scripts/**

- **eval_360VOT.py**:Used for reproducing inference benchmarks. This script performs quantitative analysis and generates attribute-based performance graphs for the 120 test set sequences.
- **vis_result_360VOT.py**:
  Utilized for qualitative analysis of the test set sequences, which helped in curating subset sequences for further experiments.

### 2. **scripts/utility_scripts/**

- **benchmark_vot_test.py**:Benchmarks the entire 360VOT test set using AiATrack with various hyperparameter configurations. It was instrumental in experimenting with different setups to address Hypothesis 2 and manage occlusions. These experiments were conducted on 3 NVIDIA Titan X GPUs in a data-parallel setting.
- **generate_figs.py**:Creates mosaic images to explain attributes in the report and generates attribute-based performance graphs for the top-performing trackers.
- **reproduce_bbox_results.ipynb**:
  Visualizes bounding box outputs for AiATrack and AiATrack-360 using inference results from 360VOT.

### 3. **scripts/**

- **bfov_vis_aiatrack.ipynb**:A data analysis notebook that visualizes different local search regions per frame to analyze tracker input after integrating the 360 tracking framework. It helps identify where the tracker begins to drift. This notebook leverages the `OmniImage` class from `360VOT/lib/omni.py`, which defines the tracking framework and includes helper functions for various projection methods.
- **data_analysis.ipynb**:A utility notebook used for curating edge cases based on attributes and developing class activation maps to investigate potential issues with distortion in perspective CNNs.
- **erp_ops.ipynb**:Explores rectilinear projection, which informed the development of the tangent plane projector re-orientation method. This notebook was crucial in understanding how projection conversions work in 360 images.
- **pano_to_plane.ipynb**:A beta version of the tangent plane projector re-orientation method, which dynamically re-orients the camera view while projecting an ERP region onto a plane. This class is part of the Hypothesis 1 mitigation strategy.
- **SphereProjection.ipynb**:Replicates the Kernel Transformer network training pipeline, proposed as future work to evaluate its viability on the 360VOT train set. The `SphereProjection` class generates tangent plane projections across all points in the image to facilitate CNN transfer learning on these projections.
- **test_hypothesis.ipynb**:
  Integrates the tangent projector logic developed in `pano_to_plane.ipynb` with the tracker inference code to run inference across edge cases on NVIDIA Titan X GPUs.

### 4. **AiATrack/lib/test/tracker/**

- **aiatrack.py**:
  Modifications to the `track` function to incorporate the tangent projector logic for dynamic view re-orientation.

### 5. **AiATrack/lib/test/utils/**

- **TangentProjector.py**:
  A utility class used by the AiATrack `track` function to dynamically re-orient views during tracking.

## Getting Started

1. **Prerequisites**:

   - Python 3.6
   - NVIDIA GPUs (Titan X recommended)
   - Required Python packages (listed in `requirements.txt`)
2. **Running Benchmarks**:Use `benchmark_vot_test.py` to benchmark the 360VOT test set with various configurations.
3. **Visualization**:Use `reproduce_bbox_results.ipynb` and `bfov_vis_aiatrack.ipynb` for visualizing bounding box outputs and analyzing tracker behavior.
4. **Testing Hypotheses**:
   The `test_hypothesis.ipynb` notebook integrates the tangent projector logic for inference testing on edge cases.

## References

This work heavily relies on the following research and repository codes:

- Gao, Shenyuan, et al. *"AiATrack: Attention in Attention for Transformer Visual Tracking."* European Conference on Computer Vision. Springer, 2022.
- Huang, Huajian, et al. *"360VOT: A New Benchmark Dataset for Omnidirectional Visual Object Tracking."* Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- Su, Yu-Chuan, and Kristen Grauman. *"Kernel Transformer Networks for Compact Spherical Convolution."* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

## Future Work

The `SphereProjection.ipynb` notebook outlines a potential future direction involving the Kernel Transformer network training pipeline, evaluating its performance on the 360VOT train set.
