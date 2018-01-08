<img src="https://drive.google.com/uc?id=1Wa_O-8Cabvj8Dp7xT3J8KnhyfvzMF55s" align="right" width="384" />

# horse2zebra

## Training

```bash
./train.py --gpu 0
```

![](../cyclegan/.readme/horse2zebra_epoch77.jpg)


## PyTorch to Chainer

```bash
# install pytorch
conda install -c soumith -y pytorch cuda80

./download_models.sh

./infer_pytorch.py

# To check G's re-implementation
./pytorch2chainer_G.py
./infer_chainer.py

# To check D's re-implementation
./generate_D_random_pth.py
./pytorch2chainer_D.py
```

<img src=".readme/horse2zebra_pytorch.jpg" width="60%" />
<img src=".readme/horse2zebra_chainer.jpg" width="60%" />
