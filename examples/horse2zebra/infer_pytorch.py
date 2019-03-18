#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
from torch.autograd import Variable

here = osp.dirname(osp.realpath(__file__))  # NOQA
pytorch_dir = osp.join(here, '../../src/pytorch-cyclegan')  # NOQA

sys.path.insert(0, pytorch_dir)  # NOQA
from models import networks


def main():
    default_model_file = osp.join(here, 'data/G_horse2zebra.pth')
    default_img_file = 'https://images2.onionstatic.com/clickhole/3570/2/original/600.jpg'  # NOQA

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('-m', '--model-file', default=default_model_file,
                        help='Model file.')
    parser.add_argument('-i', '--img-file', default=default_img_file,
                        help='Image file.')
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print('GPU id: {:d}'.format(args.gpu))
    print('Model file: {:s}'.format(args.model_file))
    print('Image file: {:s}'.format(args.img_file))

    model = networks.ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=networks.get_norm_layer(norm_type='instance'),
        use_dropout=False,
        n_blocks=9,
        gpu_ids=[args.gpu],
        padding_type='reflect',
    )
    model.load_state_dict(torch.load(args.model_file))
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()

    img = skimage.io.imread(args.img_file)

    batch_size = 3

    img_org = img.copy()
    img = cv2.resize(img, (256, 256))

    xi = img.astype(np.float32)
    xi = (xi / 255 * 2) - 1
    xi = xi.transpose(2, 0, 1)
    x = np.repeat(xi[None, :, :, :], batch_size, axis=0)
    x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    x = Variable(x, volatile=True)

    y = model(x)

    yi = y[0].data
    yi = (yi + 1) / 2 * 255
    yi = yi.cpu().numpy()
    yi = yi.transpose(1, 2, 0)
    out = yi.astype(np.uint8)
    out = cv2.resize(out, (img_org.shape[1], img_org.shape[0]))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_org)
    plt.title('Input (PyTorch)')
    plt.subplot(122)
    plt.imshow(out)
    plt.title('Output (PyTorch)')
    plt.show()


if __name__ == '__main__':
    main()
