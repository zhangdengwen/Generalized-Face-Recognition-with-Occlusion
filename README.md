## Introdcution

Generalized Face Recognition with Occlusion (GFRO) is an occlusion-robust face recognition framework that generalizes to diverse real-world occlusions (e.g., masks, glasses, hats) without occlusion-specific training data. GFRO learns generic representations from landmark-guided partial-face views and employs a dual Mixture-of-Experts (MoE) aggregator to refine and integrate multi-branch features, aligning partial-face embeddings with clean-face embeddings via a clean–noisy contrastive loss, while maintaining identity discriminability with an ArcFace classification objective. The implementation follows the training/evaluation protocols of FaceX-Zoo for reproducibility.

## Requirements

* python >= 3.7.1

* pytorch >= 1.1.0

* torchvision >= 0.3.0

See the detail requirements in [requirements.txt](./requirements.txt)

## Model Training

run [./train.sh](./training_mode/conventional_training/train.sh) with the train dataset CASIA-Webface ([baidu Pan Link](https://pan.baidu.com/s/1mSbJ61BWEqPqv6RZkqv7CQ?pwd=877a)).

Download teacher model (ElasticFace-Arc) from [ElasticFace](https://github.com/fdbtrs/ElasticFace). 

You can switch the training data type by editing [data_processor/train_dataset.py](./data_processor/train_dataset.py)
. This file controls how training samples are constructed. By changing the dataset option, you can train with randomly cropped partial-face samples or synthetic occlusion samples (e.g., mask/glasses/hat-style occlusions).

## Model Test

run [./test_lfw_adapt.sh](./test_protocol/test_lfw_adapt.sh) with the test dataset LFW-MASK ([baidu Pan Link](https://pan.baidu.com/s/1bVmH67D1SWpgv2Fb3rg66A?pwd=p50q)), or the dataset MEGLSS([baidu Pan Link](https://pan.baidu.com/s/1r_7O0GxDkEMNkb4Kvty_9A?pwd=wg1m)), CALFW-SUNGLASSES([baidu Pan Link](https://pan.baidu.com/s/190MAC_RNLykQdmypUuQgGg?pwd=l7p1)), CPLFW([baidu Pan Link](https://pan.baidu.com/s/1gJ8659xUhG-gcOZ4fMS6XA?pwd=6wmo))

## Simulated Occlusion / Crop Generation

You can generate cropped partial-face datasets or simulated occluded datasets for evaluation by running the following scripts (please edit the dataset paths and occlusion settings inside each file before running):

- [crop_lfw_random.py](./test_protocol/lfw/crop_lfw_random.py): generate Crop/Occlusion test set for LFW
- [crop_cplfw_random.py](./test_protocol/lfw/crop_cplfw_random.py): generate Crop/Occlusion test set for CPLFW
- [crop_calfw_random.py](./test_protocol/lfw/crop_calfw_random.py): generate Crop/Occlusion test set for CALFW
