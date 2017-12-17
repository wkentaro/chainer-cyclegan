#!/usr/bin/env python

import os.path as osp
import sys

import numpy as np
import torch
from torch.autograd import Variable


here = osp.dirname(osp.realpath(__file__))
pytorch_dir = osp.join(here, 'src/pytorch-cyclegan')

sys.path.insert(0, pytorch_dir)
from models import networks


norm_layer = networks.get_norm_layer(norm_type='instance')
model = networks.ResnetGenerator(
    input_nc=3,
    output_nc=3,
    ngf=64,
    norm_layer=norm_layer,
    use_dropout=False,
    n_blocks=9,
    gpu_ids=[0],
    padding_type='reflect',
)
model_file = osp.join(here, 'data/G_horse2zebra.pth')
model.load_state_dict(torch.load(model_file))
model = model.cuda()
model = model.eval()

import skimage.io
img_file = 'https://images2.onionstatic.com/clickhole/3570/2/original/600.jpg'
img = skimage.io.imread(img_file)

xi = img.astype(np.float32)
xi = (xi / 255 * 2) - 1
xi = xi.transpose(2, 0, 1)
x = np.repeat(xi[None, :, :, :], 3, axis=0)
x = torch.from_numpy(x).cuda()
x = Variable(x, volatile=True)

y = model(x)

yi = y[0].data
yi = (yi + 1) / 2 * 255
yi = yi.cpu().numpy()
yi = yi.transpose(1, 2, 0)
out = yi.astype(np.uint8)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img)
plt.title('Input (PyTorch)')
plt.subplot(122)
plt.imshow(out)
plt.title('Output (PyTorch)')
plt.show()
