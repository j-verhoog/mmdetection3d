# Federated 3D Object Detection — Thesis & Paper Experiments

This repository contains the code, experiment configurations, and analysis scripts developed for my MSc thesis and related paper submission on **federated learning for 3D object detection under domain heterogeneity**.

The experiments are based on the **nuScenes** dataset and use three 3D object detectors:

* **CMT (Cross-Modal Transformer)**
* **PointPillars**
* **FCOS3D**

The main experiments investigate federated learning across heterogeneous driving domains, including differences in **location, time of day, and weather conditions**.

Unless stated otherwise, experiments were run on the **DAIC compute infrastructure** using Slurm.

## Repository Structure

### `analysis`

Analysis and evaluation scripts used across multiple experiments.

These scripts include utilities for processing experiment outputs, comparing methods, and generating results used in the thesis and paper.

### `cmt_40_epoch`

**Main experimental directory for the thesis and paper.**

Contains the final CMT experiments using **40 federated training rounds** and a five-domain partition of nuScenes:

1. Boston — day, clear
2. Boston — day, rain
3. Singapore — day, clear
4. Singapore — night, clear
5. Singapore — night, rain

This directory contains the primary experiments comparing federated learning and personalization methods, including the experiments used to evaluate the proposed **FedCKA** approach.

### `pointpillars_ablation`

Contains the **PointPillars ablation experiments used in the paper**.

These experiments use the same five-domain problem as the main CMT experiments and provide an additional detector architecture for evaluating whether the observed behavior generalizes beyond CMT.

### `cmt`

Contains earlier CMT experiments using **20 training rounds** and a simpler two-domain partition:

* Rain
* No rain

These experiments preceded the final five-domain experimental setup.

### `pointpillars`

Contains earlier experiments using a lightweight **PointPillars** configuration on the two-domain **rain / no-rain** problem.

### `pointpillars4x8`

Contains PointPillars experiments used to replicate the original implementation/results on the two-domain setup.

### `cmt_full`

Contains experiments used to reproduce the results of the original **CMT** implementation before applying it to the federated learning experiments.

### `fcos3d`

Contains experiments involving the **FCOS3D** detector.

These experiments were part of the broader exploration of detector architectures during the project.

### `cmt_delftblue`

Contains CMT experiments run on the **DelftBlue** compute cluster instead of DAIC.

### `subsets_creation`

Contains scripts used to construct the nuScenes domain subsets used in the federated learning experiments.

These scripts define the dataset partitions based on properties such as location, weather conditions, and time of day.

### `setup`

Contains environment and infrastructure setup files, including:

* DAIC configuration
* Slurm scripts
* Environment setup
* Dataset preparation utilities
* Legacy dataset setup scripts

## Main Experimental Setup

The final experiments model each nuScenes domain as a separate federated client.

The primary five-client setup consists of:

| Client | Domain                   |
| ------ | ------------------------ |
| A      | Boston — day, clear      |
| B      | Boston — day, rain       |
| C      | Singapore — day, clear   |
| D      | Singapore — night, clear |
| E      | Singapore — night, rain  |

The main experiments use **CMT** as the detector, while **PointPillars** is used as an additional architecture for ablation and generalization experiments.

## Research Context

This repository accompanies research on **generalization and personalization in federated 3D object detection**.

The work investigates how model parameters should be shared between clients when the local data distributions differ substantially. In particular, the experiments study domain shifts caused by differences in:

* Geographic location
* Day and night conditions
* Clear and rainy weather

The proposed method, **FedCKA**, dynamically determines which parts of the detector should remain personalized based on representation similarity between local and global models.

## Notes

Some directories contain exploratory, replication, or legacy experiments that were conducted during the development of the final experimental setup.

For the experiments most directly corresponding to the final thesis and paper, see:

* `cmt_40_epoch`
* `pointpillars_ablation`
* `analysis`
* `subsets_creation`
