#!/usr/bin/env python
# flake8: noqa

import os.path as osp
import sys

import chainer
import chainer.links as L
import torch
import numpy as np

from chainer_cyclegan.links import InstanceNormalization
from chainer_cyclegan.models import NLayerDiscriminator


here = osp.dirname(osp.realpath(__file__))

model_file = osp.join(here, 'data/D_random.pth')
state_dict = torch.load(model_file)

params = []
for k, v in state_dict.items():
    print(k)
    if 'running' not in k:
        params.append(v.numpy().flatten())
params = np.hstack(params)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('D original (PyTorch)')
print(params.size, params.min(), params.mean(), params.max())
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

model = NLayerDiscriminator()

params = []
for param in model.params():
    params.append(param.array.flatten())
params = np.hstack(params)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('D init (Chainer)')
print(params.size, params.min(), params.mean(), params.max())
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
for i, func in enumerate(model.functions):
    if isinstance(func, L.Convolution2D):
        np.copyto(func.W.array, state_dict['model.{:d}.weight'.format(i)].numpy())
        np.copyto(func.b.array, state_dict['model.{:d}.bias'.format(i)].numpy())
    elif isinstance(func, InstanceNormalization):
        np.copyto(func.avg_mean, state_dict['model.{:d}.running_mean'.format(i)].numpy())
        np.copyto(func.avg_var, state_dict['model.{:d}.running_var'.format(i)].numpy())

params = []
for param in model.params():
    params.append(param.array.flatten())
params = np.hstack(params)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('D copied (Chainer)')
print(params.size, params.min(), params.mean(), params.max())
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

model_file = osp.join(here, 'data/D_random_from_pytorch.npz')
chainer.serializers.save_npz(model_file, model)
print('Saved model file: {:s}'.format(model_file))
