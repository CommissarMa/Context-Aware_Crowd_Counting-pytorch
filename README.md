# CSRNet-Simple-Pytorch
This is an simple and clean implemention of CVPR 2018 paper ["CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"](https://arxiv.org/abs/1802.10062).  
# Installation
&emsp;1. Install pytorch  
&emsp;2. Install visdom    
```pip
pip install visdom
```
&emsp;3. Install tqdm
```pip
pip install tqdm
```  
&emsp;4. Clone this repository  
```git
git clone https://github.com/CommissarMa/CSRNet-pytorch.git
```
We'll call the directory that you cloned CSRNet-pytorch as ROOT.
# Data Setup
&emsp;1. Download ShanghaiTech Dataset from
Dropbox: [link](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or Baidu Disk: [link](http://pan.baidu.com/s/1nuAYslz)  
&emsp;2. Put ShanghaiTech Dataset in ROOT and use "data_preparation/k_nearest_gaussian_kernel.py" to generate ground truth density-map. (Mind that you need modify the root_path in the main function of "data_preparation/k_nearest_gaussian_kernel.py")  
# Training
&emsp;1. Modify the root path in "train.py" according to your dataset position.  
&emsp;2. Run train.py
# Testing
&emsp;1. Modify the root path in "test.py" according to your dataset position.  
&emsp;2. Run test.py for calculate MAE of test images or just show an estimated density-map.  
# Other notes
&emsp;1. We trained the model and got 67.74 MAE at 124-th epoch which is comparable to the result of original paper. 
&emsp;2. If you are new to crowd counting, we recommand you to know [Crowd_counting_from_scratch](https://github.com/CommissarMa/Crowd_counting_from_scratch) first. It is an overview and tutorial of crowd counting.