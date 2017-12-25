#!/usr/bin/env python

import argparse
import glob
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir')
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.log_dir, 'log.csv'))

    plt.figure(dpi=100, figsize=(16, 8))

    ax = plt.subplot(231)
    df.plot(x='iteration', y=['loss_G'], ax=ax)
    plt.legend()

    ax = plt.subplot(232)
    df.plot(x='iteration', y=['loss_G_A', 'loss_G_B'], ax=ax)
    plt.legend()

    ax = plt.subplot(233)
    df.plot(x='iteration', y=['loss_idt_A', 'loss_idt_B'], ax=ax)
    plt.legend()

    ax = plt.subplot(234)
    df.plot(x='iteration', y=['loss_cycle_A', 'loss_cycle_B'], ax=ax)
    plt.legend()

    ax = plt.subplot(235)
    df.plot(x='iteration', y=['loss_D_A', 'loss_D_B'], ax=ax)
    plt.legend()

    img_files = glob.glob(osp.join(args.log_dir, '*.jpg'))
    if img_files:
        img_file = sorted(img_files, reverse=True)[0]
        plt.subplot(236)
        plt.imshow(plt.imread(img_file))
        plt.title(osp.basename(img_file))

    plt.show()


if __name__ == '__main__':
    main()
