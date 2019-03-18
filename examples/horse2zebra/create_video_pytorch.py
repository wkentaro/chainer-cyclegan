#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys

import cv2
import imageio
import numpy as np
import torch
from torch.autograd import Variable
import tqdm

here = osp.dirname(osp.realpath(__file__))  # NOQA
pytorch_dir = osp.join(here, '../../src/pytorch-cyclegan')  # NOQA

sys.path.insert(0, pytorch_dir)  # NOQA
from models import networks


def main():
    default_model_file = osp.join(here, 'data/G_horse2zebra.pth')
    default_out_file = osp.join(here, 'logs/create_horse2zebra_pytorch.gif')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_file', help='Video file of horse.')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id.')
    parser.add_argument('-m', '--model-file', default=default_model_file,
                        help='Model file.')
    parser.add_argument('-o', '--out-file', default=default_out_file,
                        help='Output video file.')
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print('GPU id: {:d}'.format(args.gpu))
    print('Model file: {:s}'.format(args.model_file))
    print('Video file: {:s}'.format(args.video_file))
    print('Output file: {:s}'.format(args.out_file))

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

    batch_size = 1

    video = imageio.get_reader(args.video_file)
    writer = imageio.get_writer(args.out_file)
    for img in tqdm.tqdm(video):
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

        writer.append_data(np.hstack([img_org, out]))

    print('Wrote video: {:s}'.format(args.out_file))


if __name__ == '__main__':
    main()
