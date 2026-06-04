# MMCFM-PNDA

This repository implements Multimodal-Conditioned Flow Matching for Pathology Nuclei Data Augmentation.

# Data preparation
Multi-class pathology nuclei dataset preparation
Sel-supervised learning (SSL) feature representation
Pathology foundation model: UNI, Virchow2,....

Text descriptions of multi-class nuclei mask
"Organ tissue type, nulcei type: proporation"
Pathology visual-language model: QuiltNet, MUSK, ....

Quick start

- Train image model:
  python train_image.py --config_path configs/PanNuke.yaml

- Train mask model:
  python train_mask.py --config_path configs/PanNuke_Mask.yaml

Project layout

- train.py: unified training entry (used by train_image.py and train_mask.py)
- train_image.py / train_mask.py: thin CLI wrappers
- utils.py: datasets, model creation, checkpointing and validation utilities
- models/: model definitions (ControlNet, UNet)
- configs/: YAML configs for image / mask training
