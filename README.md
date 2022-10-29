# Multitask-Learning-Bridge-Inspection
This is the implementation of the paper ["A Multitask Deep Learning Model for Parsing Bridge Elements and Segmenting Defect in Bridge Inspection Images"](https://arxiv.org/abs/2209.02190).

## Network Architecture
![MTL-architecture](https://user-images.githubusercontent.com/90736946/198709157-abf0d92a-1b28-4459-a099-7e4ccd5b9006.png)

## Getting Started
* Install the required dependencies.
* Dataset: Please download the initial dataset from [here](https://drive.google.com/drive/folders/1HLCUC8R9x3t-qB_t3NQ1XujMV43Axmv_?usp=share_link), and then unzip it place in `./VOCdevkit/VOC2007/`.
* Pre-trained weights: Please download pre-trained weights on VOC12+SBD from [here](https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_weights_voc.pth), and then place it in `./model_data/`.
*  train_multi.py: this code is used for loading data and training the model.
*  predict_multi.py: this code is used for inferencing. 

## Citation
If this work is helpful to you, please cite it as:
~~~~
@misc{https://doi.org/10.48550/arxiv.2209.02190,
  title = {A Multitask Deep Learning Model for Parsing Bridge Elements and Segmenting Defect in Bridge Inspection Images},
  author = {Zhang, Chenyu and Karim, Muhammad Monjurul and Qin, Ruwen},
  doi = {10.48550/ARXIV.2209.02190},
  url = {https://arxiv.org/abs/2209.02190},
  publisher = {arXiv}, 
  year = {2022},
}
~~~~
Note that part of the codes are referred from <a href="https://github.com/bubbliiiing/hrnet-pytorch">HRnet-pytorch</a> project.
