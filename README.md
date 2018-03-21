<img src="https://drive.google.com/uc?id=1APsWYE6fx1a6PNrG1saiFuBHcuZOXZln" align="right" width="384" />

# chainer-cyclegan

[![Build Status](https://travis-ci.org/wkentaro/chainer-cyclegan.svg?branch=master)](https://travis-ci.org/wkentaro/chainer-cyclegan)

Chainer implementation of ["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Network"](https://arxiv.org/abs/1703.10593).  
This is a faithful re-implementation of [the official PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


## Installation

```bash
git clone --recursive https://github.com/wkentaro/chainer-cyclegan.git
cd chainer-cyclegan

conda install -c menpo -y opencv
pip install .
```

or

```bash
pip install chainer-cyclegan
```


## Training

```bash
cd examples/cyclegan

# ./train.py <dataset> --gpu <gpu_id>
./train.py horse2zebra --gpu 0
```

## Results

### apple2orange

![](examples/cyclegan/.readme/apple2orange_epoch200.jpg)

### summer2winter_yosemite

![](examples/cyclegan/.readme/summer2winter_yosemite_epoch94.jpg)

### horse2zebra

![](examples/cyclegan/.readme/horse2zebra_epoch200.jpg)

### monet2photo

![](examples/cyclegan/.readme/monet2photo_epoch50.jpg)

### cezanne2photo

![](examples/cyclegan/.readme/cezanne2photo_epoch22.jpg)

### ukiyoe2photo

![](examples/cyclegan/.readme/ukiyoe2photo_epoch108.jpg)

### vangogh2photo

![](examples/cyclegan/.readme/vangogh2photo_epoch63.jpg)

### maps

![](examples/cyclegan/.readme/maps_epoch200.jpg)

### cityscapes

![](examples/cyclegan/.readme/cityscapes_epoch17.jpg)

### facades

![](examples/cyclegan/.readme/facades_epoch200.jpg)

### iphone2dslr_flower

![](examples/cyclegan/.readme/iphone2dslr_flower_epoch200.jpg)
