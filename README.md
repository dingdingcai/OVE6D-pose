# OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation
- [Paper](https://bop.felk.cvut.cz/datasets/)

## Setup
Please start by installing [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) with Pyhton3.8 or above.

## Dataset
Our evaluation is conducted on three datasets all downloaded from [BOP website](https://bop.felk.cvut.cz/datasets). All three datasets are stored in the same directory. e.g. ``BOP_Dataset/lm, BOP_Dataset/lmo, BOP_Dataset/tless``.

## Quantitative Evaluation
The evaluation code is based on the code [bop_toolkit](https://github.com/thodan/bop_toolkit) (ADD(-S) on LINEMOD and Occluded LINEMOD) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit)(VSD on T-LESS single object per class).

Evaluation on the LineMOD and Occluded LineMOD datasets with instance segmentation (Mask-RCNN) network (complete pipeline)

``python LM_RCNN_OVE6D_pipeline.py`` for LineMOD.
``python LMO_RCNN_OVE6D_pipeline.py`` for Occluded LineMOD.

Evaluation on the T-LESS dataset with the provided object segmentation masks.

``python TLESS_eval_sixd17.py`` for TLESS.

## Training
To train DVE6D, the ShapeNet dataset is required. You must first pre-process ShapeNet with the provided script in ``training/preprocess_shapenet.py`` ., and [Blender](https://www.blender.org/) is required for this task. More details refer to [LatentFusion](https://github.com/NVlabs/latentfusion).

## pre-trained weight for OVE6D
Our pre-trained OVE6D weights can be found [here](https://drive.google.com/file/d/1aXkYOpvka5VAPYUYuHaCMp0nIvzxhW9X/view?usp=sharing).

# Segmentation Masks
- 1. For T-LESS we use the [segmentation masks](https://dlrmax.dlr.de/get/c677b2a7-78cf-5787-815b-7ba2c26555a7/) provided by [Multi-Path Encoder](https://github.com/DLR-RM/AugmentedAutoencoder/tree/multipath).
- 2. For LineMOD and Occluded LineMOD, we fine-tuned the Mask-RCNN initialized with the weights from [Detectron2](https://github.com/facebookresearch/detectron2). The training data can be downloaded from [BOP](https://bop.felk.cvut.cz/datasets). Our pretrained Mask-RCNN weights can be downloaded from [LM](https://drive.google.com/file/d/1AEV5XO975RYiPXjWSDQ1TsmW-BalbfMU/view?usp=sharing) and [LMO](https://drive.google.com/file/d/1tut-wZyi1RQ52c65ZfBtM2k9snDVMk_l/view?usp=sharing). 

# Acknowledgement
- 1. The code is partly borrowed from [LatentFusion](https://github.com/NVlabs/latentfusion).
- 2. The evaluation code is based on [bop_toolkit](https://github.com/thodan/bop_toolkit) and [sixd_toolkit](https://github.com/thodan/sixd_toolkit).