#  Image-Segmentation

## Introduction

The master branch works with **PyTorch C++ 1.0**. If you would like to use PyTorch 0.4.1,
please checkout to the [PyTorch C++ 1.0](https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip) branch.

Image-Segmentaion is an open source image segementation toolbox based on PyTorch Cpp.

### Major features

- **Modular Design**

  One can easily construct a customized object detection framework by combining different components.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular segmentation frameworks, *e.g.* FCN, SegNet, PspNet,DeeplabV3,Unet etc.

- **Efficient**

  All basic operations run on GPUs now.
  

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Updates

v0.1(2019/4/1)
- support FCN with backbone VGG16
- support Unet with backbone Resnet50
- support Deeplabv3 With backbone Inception
- support PsPnet with backbone Resnet50
- support Segnet with backbone Vgg19


## Benchmark and model zoo

Supported methods and backbones are shown in the below table.

|                    | ResNet   | VGG16    | Inception| VGG19    |
|--------------------|:--------:|:--------:|:--------:|:--------:|
| FCN                | ✗        | ✗        | ✗        | ✓        |
| Unet               | ✓        | ✗        | ✗        | ✗        |
| Deeplabv3          | ✗        | ✗        | ✓        | ✗        |
| PsPnet             | ✓        | ✗        | ✗        | ✗        |
| Segnet             | ✗        | ✗        | ✗        | ✓        |



## Installation

Please refer to [INSTALL](https://pytorch.org) for installation.


## Train a model

* Config CMakeLists.txt
* Compile CMakeLists.txt with make

### Train on custom datasets
* Put your training data under data folder and compile, and then you can train you data.