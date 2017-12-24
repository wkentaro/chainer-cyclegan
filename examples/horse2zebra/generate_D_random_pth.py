#!/usr/bin/env python

import os.path as osp
import sys

import torch

here = osp.dirname(osp.realpath(__file__))
pytorch_dir = osp.join(here, '../../src/pytorch-cyclegan')

sys.path.insert(0, pytorch_dir)
from models import networks


def main():
    model = networks.NLayerDiscriminator(
        input_nc=3,
        ndf=64,
        norm_layer=networks.get_norm_layer(norm_type='instance'),
        use_sigmoid=False,
        gpu_ids=[0],
    )
    out_file = osp.join(here, 'data/D_random.pth')
    torch.save(model.state_dict(), out_file)
    print('Saved model file: {:s}'.format(out_file))


if __name__ == '__main__':
    main()
