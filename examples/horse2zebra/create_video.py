#!/usr/bin/env python

import argparse
import os.path as osp

import chainer
from chainer import cuda
import numpy as np
try:
    import imageio
    import tqdm
except ImportError:
    print('Please install followings:')
    print('  pip install imageio')
    print('  pip install tqdm')
    quit(1)

import chainer_cyclegan


here = osp.dirname(osp.realpath(__file__))


def main():
    default_model_file = osp.join(here, 'data/G_horse2zebra_from_pytorch.npz')
    default_out_file = osp.join(here, 'logs/create_horse2zebra.gif')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_file', help='Video file of horse.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('-m', '--model-file', default=default_model_file,
                        help='Model file.')
    parser.add_argument('-o', '--out-file', default=default_out_file,
                        help='Output video file.')
    args = parser.parse_args()

    print('GPU id: {:d}'.format(args.gpu))
    print('Model file: {:s}'.format(args.model_file))
    print('Video file: {:s}'.format(args.video_file))
    print('Output file: {:s}'.format(args.out_file))

    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    model = chainer_cyclegan.models.ResnetGenerator()
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    batch_size = 1

    video = imageio.get_reader(args.video_file)
    writer = imageio.get_writer(args.out_file)
    for img in tqdm.tqdm(video):
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

        writer.append_data(np.hstack([img, out]))

    print('Wrote video: {:s}'.format(args.out_file))


if __name__ == '__main__':
    main()
