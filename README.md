# 🌎🛰️ MultiMAE meets Earth Observation: Pre-training Multi-modal multi-task Masked Autoencoders for Earth Observation Tasks


## Updates
- **June 7th, 2025:** Code for pre-training our model released.
- **May 22nd, 2025:** Pre-print available [arXiv](https://arxiv.org/pdf/2505.14951).
- **May 20th, 2025:** Our paper has been acepted at [ICIP 2025](https://cmsworkshops.com/ICIP2025/papers/accepted_papers.php).

## Overview
Multi-modal data in Earth Observation (EO) presents a huge opportunity for improving transfer learning capabilities when pre-training deep learning models. Our work proposes a flexible multi-modal, multi-task pre-training strategy for EO data. Specifically, we adopt a Multi-modal Multi-task Masked Autoencoder (MultiMAE) that we pre-train by reconstructing diverse input modalities, including spectral, elevation, and segmentation data. The pre-trained model demonstrates robust transfer learning capabilities, outperforming state-of-the-art methods on various EO datasets for classification and segmentation tasks. Our approach exhibits significant flexibility, handling diverse input configurations without requiring modality-specific pre-trained models.

## Approach

<img width="1096" alt="image" src="images/main_arch.png">

## Set-up
1. Clone this repo: `https://github.com/josesosajs/multimae-meets-eo.git`
2. Create a conda environment from the configuration file: `environment_config.yml`

## Pre-training

### Data
Our model is pre-trained with [MMEarth](https://github.com/vishalned/MMEarth-data) dataset. Follow the instructions in their [repo](https://github.com/vishalned/MMEarth-data/blob/main/README.md) to download the data.

All their datasets have a similar structure: 

    .
    ├── data_1M_v001/                      # root data directory
    │   ├── data_1M_v001.h5                # h5 file containing the data
    │   ├── data_1M_v001_band_stats.json   # json file containing information about the bands present in the h5 file for each data stack
    │   ├── data_1M_v001_splits.json       # json file containing information for train, val, test splits
    │   └── data_1M_v001_tile_info.json    # json file containing additional meta information of each tile that was downloaded. 

### Changing configuration
The pre-training script reads from a default configuration file located [here](cfgs/pretrain/). Change the configuration values on the file according to your needs. Alternatively, you can add the arguments directly on the command.

Update `file_path` variable [here](utils/data_constants.py), with the path to the file with the dataset stats, e.g. `data_1M_v001_band_stats.json` .

### Logs
Our implementation includes support for logging pre-training metrics to [Weights & Biases](https://wandb.ai/site/). Make sure to set your wandb account, then add the corresponding values to the config file, or deactivate this option using `--log_wandb False`.

### Start pre-training
Run the following command to pre-train our MultiMAE implementation with [multi-modal EO data](https://github.com/vishalned/MMEarth-data) on 4 gpus:

```bash
torchrun --nproc_per_node=4 run_pretraining.py \
--config cfgs/pretrain/default_config.yaml \
--data_path path_to_mmearth_data/data_1M_v001.h5
```

## Fine-tuning
Code coming soon. Stay tuned ...


## Acknowledgements
Our work is inspired by [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) and it uses [MMEarth](https://github.com/vishalned/MMEarth-data) dataset. We thank authors for making their amazing work accesible to the community. 

## Citation
```
@article{sosa2025multimae,
  title={MultiMAE Meets Earth Observation: Pre-training Multi-modal Multi-task Masked Autoencoders for Earth Observation Tasks},
  author={Sosa, Jose and Rukhovich, Danila and Kacem, Anis and Aouada, Djamila},
  journal={arXiv preprint arXiv:2505.14951},
  year={2025}
}
```