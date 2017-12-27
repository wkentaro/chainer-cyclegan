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

![](examples/cyclegan/.readme/apple2orange_epoch112.jpg)

### summer2winter_yosemite

![](examples/cyclegan/.readme/summer2winter_yosemite_epoch94.jpg)

### horse2zebra

![](examples/cyclegan/.readme/horse2zebra_epoch77.jpg)

### monet2photo

![](examples/cyclegan/.readme/monet2photo_epoch8.jpg)

### cezanne2photo

![](examples/cyclegan/.readme/cezanne2photo_epoch18.jpg)

<!--
### ukiyoe2photo

![](examples/cyclegan/.readme/ukiyoe2photo.jpg)
-->

### vangogh2photo

![](examples/cyclegan/.readme/vangogh2photo_epoch17.jpg)

### maps

![](examples/cyclegan/.readme/maps_epoch42.jpg)

### cityscapes

![](examples/cyclegan/.readme/cityscapes_epoch17.jpg)

### facades

![](examples/cyclegan/.readme/facades_epoch8.jpg)

### iphone2dslr_flower

![](examples/cyclegan/.readme/iphone2dslr_flower_epoch32.jpg)
