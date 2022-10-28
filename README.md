# Multitask-Learning-Bridge-Inspection
This is the implementation of the paper ["A Multitask Deep Learning Model for Parsing Bridge Elements and Segmenting Defect in Bridge Inspection Images"](https://arxiv.org/abs/2209.02190).

## Network Architecture
![MTL-architecture](https://user-images.githubusercontent.com/90736946/198709157-abf0d92a-1b28-4459-a099-7e4ccd5b9006.png)

## Getting Started
* Install the required dependencies: (for reference see [how_to_install.pdf](https://github.com/monjurulkarim/Tracking_manufacturing/blob/master/how_to_install.pdf) )
* [Dataset](https://drive.google.com/drive/folders/1vTNgPi2SSefO9fzxHxCa2Sgmec_B8MkM?usp=sharing): Download the initial dataset from here.
* [Coco_weight](https://drive.google.com/drive/folders/1wYTNf4nf_79OgqTSOcVc39XVr6s4d9Z-?usp=sharing): Download pre-trained resnet_50 coco weights from here.
* [weights](https://drive.google.com/drive/folders/1wYTNf4nf_79OgqTSOcVc39XVr6s4d9Z-?usp=sharing): Download pre-trained resnet_50 coco weights and trained weights for bridge element segmentation from here.
*  [custom.py](https://github.com/monjurulkarim/active_learning/blob/main/custom.py) : this code is used for loading data and training the model
*  [Training.ipynb](https://github.com/monjurulkarim/active_learning/blob/main/Training.ipynb): loading the weight and calling the training function
*  [inference.ipynb](https://github.com/monjurulkarim/active_learning/blob/main/inference.ipynb): this code is used for inferencing. 
*  [mrcnn/visualize.py](https://github.com/monjurulkarim/active_learning/blob/main/mrcnn/visualize.py) : this code is used for visualizing the segmented bridge elements with mask.

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
Note that part of the codes are referred from <a href="https://github.com/matterport/Mask_RCNN">Mask RCNN</a> project.
