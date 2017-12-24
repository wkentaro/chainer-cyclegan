# chainer-cyclegan

Chainer implementation of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Install

```bash
make install
```


## Horse2Zebra

```bash
cd examples/horse2zebra
```

### PyTorch to Chainer

```bash
./download_models.sh

./infer_pytorch.py
./infer_chainer.py
```

<img src=".readme/horse2zebra_pytorch.jpg" width="60%" />
<img src=".readme/horse2zebra_chainer.jpg" width="60%" />

### Training

```bash
./train.py --gpu 0
```
