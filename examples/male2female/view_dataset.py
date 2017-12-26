#!/usr/bin/env python

import argparse

import cv2
import fcn

from chainer_cyclegan.datasets import Male2FemaleDataset


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--split', default='train',
                        choices=['train', 'test'], help='Split of dataset.')
    args = parser.parse_args()

    print('# Parameters')
    for key, value in args.__dict__.items():
        print('{}: {}'.format(key, value))
    print('')

    dataset = Male2FemaleDataset(args.split)

    index = 0
    while True:
        img_A, img_B = dataset[index]
        print('[{:08}] img_A: {}, img_B: {}'
              .format(index, img_A.shape, img_B.shape))

        viz = fcn.utils.get_tile_image([img_A, img_B], (1, 2))
        cv2.imshow(__file__, viz[:, :, ::-1])

        key = cv2.waitKey(0)
        if key == ord('n'):
            index = max(0, index + 1)
        elif key == ord('p'):
            index = min(len(dataset), index - 1)
        elif key == ord('q'):
            break
        else:
            pass


if __name__ == '__main__':
    main()
