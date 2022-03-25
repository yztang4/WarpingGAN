# WarpingGAN
The pytorch implementation of CVPR 2022 paper: "WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation".

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