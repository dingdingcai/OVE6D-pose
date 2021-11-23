# OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation
- [Paper](https://bop.felk.cvut.cz/datasets/)

## Setup
Please start by installing [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) with Pyhton3.8 or above.

## Dataset
Our evaluation is conducted on three datasets all downloaded from [BOP website](https://bop.felk.cvut.cz/datasets). All three datasets are stored in the same directory. e.g. ``BOP_Dataset/lm, BOP_Dataset/lmo, BOP_Dataset/tless``.

## Quantitative Evaluation
The evaluation code is based on the code [bop_toolkit](https://github.com/thodan/bop_toolkit) (ADD(-S) on LINEMOD and Occluded LINEMOD) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit)(VSD on T-LESS single object per class).

``python LM_RCNN_OVE6D_pipeline.py`` for LineMOD dataset.

``python LMO_RCNN_OVE6D_pipeline.py`` for LineMOD-Occlusion dataset.

``python TLESS_eval_sixd17.py`` for TLESS dataset.

## Training
To train DVE6D, the ShapeNet dataset is required. You must first pre-process ShapeNet with the provided script in ``training/preprocess_shapenet.py`` ., and [Blender](https://www.blender.org/) is required for this task. More details refer to [LatentFusion](https://github.com/NVlabs/latentfusion).


# Acknowledgement
- 1. The code is partly borrowed from [LatentFusion](https://github.com/NVlabs/latentfusion).
- 2. The evaluation code is based on [bop_toolkit](https://github.com/thodan/bop_toolkit) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit).