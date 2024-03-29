# WarpingGAN
The pytorch implementation of CVPR 2022 paper: "WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation".
<img width="1693" alt="postercvpr" src="https://user-images.githubusercontent.com/101309618/171564261-f9925b93-985f-4ddc-86cc-f6b8a9988330.png">

Introduce Video https://youtu.be/3x0UckQHCDg

The generated chair and airplane samples (for visualization or evaluation), pretrained models, and inference file are available here: [https://drive.google.com/file/d/1YwxoyiWLJC4gxVgZ-ntijo9NQEC0MRlW/view](https://drive.google.com/drive/folders/11QLbw35M_49VoUvYdi803R1T8CccpDF4?usp=sharing). 

### Usage

1. requires:

   ```
   CUDA10 + Pytorch 1.4 + Python3 + Visdom
   ```

2. Dataset:

   ```
   download data: https://github.com/charlesq34/pointnet-autoencoder#download-data
   shapenetcore_partanno_segmentation_benchmark_v0
   ```

3. Setting
   ```
   In our project, arguments.py file has almost every parameters to specify for training. (We implement the project based on TreeGAN's code.)

   For example, if you want to train, it needs to specify dataset_path argument.
   ```

4. Train:
   ```
   python -m visdom.server
   python train.py
   ```
