# Multitask-Learning-Bridge-Inspection
This is the implementation of the paper ["A Multitask Deep Learning Model for Parsing Bridge Elements and Segmenting Defect in Bridge Inspection Images"](https://arxiv.org/abs/2209.02190).

## Network Architecture
![MTL-architecture](https://user-images.githubusercontent.com/90736946/198709157-abf0d92a-1b28-4459-a099-7e4ccd5b9006.png)

## Getting Started
### Installation
* Install the required dependencies in `requirements.txt`.
* Clone this repo:
~~~~
git clone https://github.com/itschenyu/Multitask-Learning-Bridge-Inspection.git
cd Multitask-Learning-Bridge-Inspection
~~~~
### Dataset
* Please download the initial dataset from [here](https://drive.google.com/drive/folders/1HLCUC8R9x3t-qB_t3NQ1XujMV43Axmv_?usp=share_link), and then unzip it place in `./VOCdevkit/VOC2007/`.

### Pre-trained weights
* Please download pre-trained weights on VOC12+SBD from [here](https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_weights_voc.pth), and then place it in `./model_data/`.

### Training
Train MTL-D:
~~~~
python train_MTL-D.py
~~~~
Train MTL-I:
~~~~
python train_MTL-I.py
~~~~

### Testing
Evaluating the model on the validation dataset:
~~~~
python get_miou.py
~~~~

### Inference
Place the inference images in `./img/`, and then run:
~~~~
python predict.py
~~~~

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
## Note
Part of the codes are referred from <a href="https://github.com/bubbliiiing/hrnet-pytorch">HRNet-pytorch</a> project.
The images and corrosion annotations in the dataset are credited to [Corrosion Condition State Semantic Segmentation Dataset](https://data.lib.vt.edu/articles/dataset/Corrosion_Condition_State_Semantic_Segmentation_Dataset/16624663/2).
