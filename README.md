# chainer-cyclegan

Chainer implementation of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Install

```bash
make install
```

## Convert PyTorch Model to Chainer

```bash
make pytorch2chainer
make test_pytorch2chainer
```

<img src=".readme/horse2zebra_pytorch.jpg" width="60%" />
<img src=".readme/horse2zebra_chainer.jpg" width="60%" />


## TODO

- [ ] Write updater.
- [ ] Write `train.py`.
