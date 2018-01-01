<img src="examples/horse2zebra/.readme/horse2zebra.gif" align="right" width="384" />

# chainer-cyclegan

Chainer implementation of ["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Network"](https://arxiv.org/abs/1703.10593).  
This is a faithful re-implementation of [the official PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


## Installation

```bash
git clone --recursive https://github.com/wkentaro/chainer-cyclegan.git
cd chainer-cyclegan

conda install -c menpo -y opencv
pip install .
```


## Training

```bash
cd examples/cyclegan

# ./train.py <dataset> --gpu <gpu_id>
./train.py horse2zebra --gpu 0
```

### apple2orange

![](.readme/apple2orange_epoch200.jpg)

### summer2winter_yosemite

![](.readme/summer2winter_yosemite_epoch94.jpg)

### horse2zebra

![](.readme/horse2zebra_epoch200.jpg)

### monet2photo

![](.readme/monet2photo_epoch50.jpg)

### cezanne2photo

![](.readme/cezanne2photo_epoch22.jpg)

### ukiyoe2photo

![](.readme/ukiyoe2photo_epoch108.jpg)

### vangogh2photo

![](.readme/vangogh2photo_epoch63.jpg)

### maps

![](.readme/maps_epoch200.jpg)

### cityscapes

![](.readme/cityscapes_epoch17.jpg)

### facades

![](.readme/facades_epoch200.jpg)

### iphone2dslr_flower

![](.readme/iphone2dslr_flower_epoch200.jpg)
