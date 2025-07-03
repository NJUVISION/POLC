# \[ICME 2025 (Oral)\] Perception-Oriented Latent Coding for High-Performance Compressed Domain Semantic Inference


## Introduction
This repository is the offical PyTorch implementation of [Perception-Oriented Latent Coding for High-Performance Compressed Domain Semantic Inference (ICME 2025)](https://arxiv.org/abs/2507.01608).

**Abstract:**
In recent years, compressed domain semantic inference has primarily relied on learned image coding models optimized for mean squared error (MSE). However, MSE-oriented optimization tends to yield latent spaces with limited semantic richness, which hinders effective semantic inference in downstream tasks. Moreover, achieving high performance with these models often requires fine-tuning the entire vision model, which is computationally intensive, especially for large models. To address these problems, we introduce Perception-Oriented Latent Coding (POLC), an approach that enriches the semantic content of latent features for high-performance compressed domain semantic inference. With the semantically rich latent space, POLC requires only a plug-and-play adapter for fine-tuning, significantly reducing the parameter count compared to previous MSE-oriented methods. Experimental results demonstrate that POLC achieves rate-perception performance comparable to state-of-the-art generative image coding methods while markedly enhancing performance in vision tasks, with minimal fine-tuning overhead.


## Preparation
The experiments were conducted on a single NVIDIA RTX A6000 with PyTorch 2.6.0, CUDA 12.6 and CuDNN9 (in the [docker environment](https://hub.docker.com/layers/pytorch/pytorch/2.6.0-cuda12.6-cudnn9-devel/images/sha256-faa67ebc9c9733bf35b7dae3f8640f5b4560fd7f2e43c72984658d63625e4487)). Create the environment, clone the project and then run the following code to complete the setup:
```bash
apt update
apt install libgl1-mesa-dev ffmpeg libsm6 libxext6 # for opencv-python
git clone https://github.com/NJUVISION/POLC.git
cd MPA
pip install -U pip && pip install -e .
```


## Pretrained Models
The trained weights after each step can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1SOb2FDsw07Za88apRk-O8diODQ6S-wkf?usp=share_link) and [Baidu Drive (access code: s83c)](https://pan.baidu.com/s/1xO-7ZdhmaeKB08nYfxMFDQ).


## Training
The training is completed by the following steps:

Step1: Run the script for variable-rate compression without GAN training pipeline:
```bash
python examples/train_stage1_wo_gan.py -m tinylic_vr -d /path/to/dataset/ --epochs 400 -lr 1e-4 --batch_size 8 --cuda --save
```

Step2: Run the script for variable-rate compression with GAN training pipeline:
```bash
python examples/train_stage1_w_gan.py -m tinylic_vr -d /path/to/dataset/ --epochs 400 -lr 1e-4 -lrd 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step1/checkpoint.pth.tar
```

Step3: Run the script for compressed-domain semantic inference training pipeline:
```bash
# for classification
# uncomment the desired cls_wrapper in the import section to train the corresponding classification model.
# change the cls model name in the wrapper train the corresponding model size.
python examples/train_stage2_cls.py -m tinylic_vr -d /path/to/imagenet-1k/ --epochs 4 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar --tag cls_perc_ft_adapter_convnext_tiny

# for semantic segmentation
python examples/train_stage2_seg.py -m tinylic_vr -a psp -d /path/to/ade20k/ --epochs 200 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar --tag seg_perc_ft_adapter_pspnet50
```

The training checkpoints will be generated in the "checkpoints" folder at the current directory. You can change the default folder by modifying the function "init()" in "expample/train.py".

For semantic segmentation, please download the checkpoint of PSPNet from [the official repo](https://github.com/hszhao/semseg) first, and save it to `checkpoints/pspnet/pspnet_train_epoch_100.pth`.


## Testing
An example to evaluate R-D performance:
```bash
# base model
python -m compressai.utils.eval_var_model checkpoint /path/to/dataset/ -a tinylic_vr -p ./path/to/step2/checkpoint.pth.tar --cuda --save /path/to/save_dir/
```

An example to evaluate classification performance:
```bash
# uncomment the desired cls_wrapper in the import section to test the corresponding classification model.
# change the cls model name in the wrapper test the corresponding model size.
python examples/train_stage2_cls.py -m tinylic_vr -d /path/to/imagenet-1k/ --epochs 4 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step3/checkpoint.pth.tar --tag cls_perc_ft_adapter_convnext_tiny --eval_only
```

An example to evaluate semantic segmentation performance:
```bash
python examples/train_stage2_seg.py -m tinylic_vr -a psp -d /path/to/ade20k/ --epochs 200 -lr 1e-4 --batch_size 8 --cuda --save --pretrained /path/to/step2/checkpoint.pth.tar --tag seg_perc_ft_adapter_pspnet50 --eval_only
```


## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{zhang2025polc,
    author = {Zhang, Xu and Lu, Ming and Chen, Yan and Ma, Zhan},
    booktitle = {2025 IEEE International Conference on Multimedia and Expo (ICME)},
    publisher = {IEEE},
    title = {Perception-Oriented Latent Coding for High-Performance Compressed Domain Semantic Inference},
    year = {2025}
}
```


## Acknowledgements
Our code is based on [MPA](https://github.com/NJUVISION/MPA), [TinyLIC](https://github.com/lumingzzz/TinyLIC), [CompressAI](https://github.com/InterDigitalInc/CompressAI), [NATTEN](https://github.com/SHI-Labs/NATTEN), [DynamicViT](https://github.com/raoyongming/DynamicViT), [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [DeiT](https://github.com/facebookresearch/deit) and [PSPNet](https://github.com/hszhao/semseg). The open-sourced baselines in our paper are reproduced from their official repositories, including [TransTIC](https://github.com/NYCU-MAPL/TransTIC) and [Adapt-ICMH](https://github.com/qingshi9974/ECCV2024-AdpatICMH). We would like to acknowledge the valuable contributions of the authors for their outstanding works and the availability of their open-source codes, which significantly benefited our work.


If you're interested in visual coding for machine, you can check out the following work from us:

- [\[NeurIPS 2024\] All-in-One Image Coding for Joint Human-Machine Vision with Multi-Path Aggregation](https://github.com/NJUVISION/MPA)
