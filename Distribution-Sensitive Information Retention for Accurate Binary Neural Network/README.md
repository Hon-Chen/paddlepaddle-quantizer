
## Getting Started

## Dependencies

* Python >= 3.6

### Installation

### Install PaddleNLP 和 Paddle

```shell
pip install paddlenlp
pip install paddlepaddle_gpu
```

### Set the paths of datasets for testing
1. Set the "dataPath" in "cifar100_resnet20.hocon" as the path root of your CIFAR-100 dataset. For example:

        dataPath = "/home/datasets/Datasets/cifar"

2. Set the "dataPath" in "imagenet_resnet18.hocon" as the path root of your ImageNet dataset. For example:

        dataPath = "/home/datasets/Datasets/imagenet"

### Training

To quantize the pretrained ResNet-20 on CIFAR-100 to 4-bit:

    python main.py --conf_path=./cifar100_resnet20.hocon --id=01
To quantize the pretrained ResNet-18 on ImageNet to 4-bit:

    python main.py --conf_path=./imagenet_resnet18.hocon --id=01

## Results

|  Dataset | Model | Pretrain Top1 Acc(%) | W4A4(Ours) Top1 Acc(%) |
   | :-: | :-: | :-: | :-: |
  | CIFAR-100 | ResNet-20| 70.33 | 63.58 ± 0.23 |
  | ImageNet | ResNet-18 | 71.47 | 60.60 ± 0.15 |

