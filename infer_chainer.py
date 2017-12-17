#!/usr/bin/env python

import os.path as osp

import chainer
from chainer import cuda
import numpy as np

from models import ResnetGenerator


here = osp.dirname(osp.realpath(__file__))

chainer.global_config.train = False
chainer.global_config.enable_backprop = False

model = ResnetGenerator()
model_file = osp.join(here, 'data/G_horse2zebra_from_pytorch.npz')
chainer.serializers.load_npz(model_file, model)
model = model.to_gpu()

import skimage.io
img_file = 'https://images2.onionstatic.com/clickhole/3570/2/original/600.jpg'
img = skimage.io.imread(img_file)

xi = img.astype(np.float32)
xi = (xi / 255 * 2) - 1
xi = xi.transpose(2, 0, 1)
x = np.repeat(xi[None, :, :, :], 3, axis=0)
x = cuda.to_gpu(x)

y = model(x)

yi = y[0].array
yi = (yi + 1) / 2 * 255
yi = cuda.to_cpu(yi)
yi = yi.transpose(1, 2, 0)
out = yi.astype(np.uint8)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.title('Output (Chainer)')
plt.subplot(122)
plt.imshow(out)
plt.title('Output (Chainer)')
plt.show()
