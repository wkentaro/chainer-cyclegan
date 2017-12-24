#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer import cuda
import matplotlib.pyplot as plt
import numpy as np
import skimage.io

import chainer_cyclegan


here = osp.dirname(osp.realpath(__file__))


def main():
    default_model_file = osp.join(here, 'data/G_horse2zebra_from_pytorch.npz')
    default_img_file = 'https://images2.onionstatic.com/clickhole/3570/2/original/600.jpg'  # NOQA

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('-m', '--model-file', default=default_model_file,
                        help='Model file.')
    parser.add_argument('-i', '--img-file', default=default_img_file,
                        help='Image file.')
    args = parser.parse_args()

    print('GPU id: {:d}'.format(args.gpu))
    print('Model file: {:s}'.format(args.model_file))
    print('Image file: {:s}'.format(args.img_file))

    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    model = chainer_cyclegan.models.ResnetGenerator()
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = skimage.io.imread(args.img_file)

    batch_size = 1

    xi = img.astype(np.float32)
    xi = (xi / 255) * 2 - 1
    xi = xi.transpose(2, 0, 1)
    x = np.repeat(xi[None, :, :, :], batch_size, axis=0)
    if args.gpu >= 0:
        x = cuda.to_gpu(x)

    y = model(x)

    yi = y[0].array
    yi = cuda.to_cpu(yi)
    yi = yi.transpose(1, 2, 0)
    yi = (yi + 1) / 2 * 255
    out = yi.astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Input (Chainer)')
    plt.subplot(122)
    plt.imshow(out)
    plt.title('Output (Chainer)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
