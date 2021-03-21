
# PVT-tensorflow2
A Tensorflow2.x implementation of Pyramid Vision Transformer as described in [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/abs/2102.12122)

## Update Log
[2021-03-20] 
* Add PVT-tiny,PVT-small,PVT-medium,PVT-large. 

## Installation
###  1. Clone project
  ``` 
  git clone https://github.com/wangermeng2021/PVT-tensorflow2.git
  cd PVT-tensorflow2
  ```

###   2. Install environment
* install tesnorflow ( skip this step if it's already installed)
*     pip install -r requirements.txt

## Training
* For training on cifar10 dataset,use:
  ```
  python train.py --dataset cifar10  --model PVT-tiny --epochs 200 --batch-size 256 --img-size 32
  ```
* For training on your custom dataset,use:
  ```
  python train.py --dataset dataset/RockPaperScissor  --model PVT-tiny --epochs 200 --batch-size 128 --img-size 128 --init-lr 5e-5
  ```
## Evaluation results:

| model                  |  cifar10  |
|------------------------|-----------|
| PVT-tiny               |  0.82     |


## References
* [https://github.com/whai362/PVT](https://github.com/whai362/PVT)
