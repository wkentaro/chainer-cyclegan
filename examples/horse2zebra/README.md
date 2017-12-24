# Horse2Zebra


## Training

```bash
./train.py --gpu 0
```


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
