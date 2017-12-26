#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'

import chainer
from chainer.datasets import TransformDataset
import chainer.optimizers as O
from chainer import training
from chainer.training import extensions
import cupy as cp
import numpy as np

from chainer_cyclegan.datasets import CycleGANTransform
from chainer_cyclegan.datasets import Horse2ZebraDataset
from chainer_cyclegan.models import NLayerDiscriminator
from chainer_cyclegan.models import ResnetGenerator
from chainer_cyclegan.training.extensions import CycleGANEvaluator
from chainer_cyclegan.training.updaters import CycleGANUpdater


def train(dataset_train, dataset_test, gpu, batch_size, suffix=''):
    np.random.seed(0)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        cp.random.seed(0)

    # Model

    G_A = ResnetGenerator()
    G_B = ResnetGenerator()
    D_A = NLayerDiscriminator()
    D_B = NLayerDiscriminator()

    if gpu >= 0:
        G_A.to_gpu()
        G_B.to_gpu()
        D_A.to_gpu()
        D_B.to_gpu()

    # Optimizer

    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    optimizer_G_A = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_G_B = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D_A = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
    optimizer_D_B = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)

    optimizer_G_A.setup(G_A)
    optimizer_G_B.setup(G_B)
    optimizer_D_A.setup(D_A)
    optimizer_D_B.setup(D_B)

    # Dataset

    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=batch_size)
    iter_test = chainer.iterators.SerialIterator(
        dataset_test, batch_size=batch_size, repeat=False, shuffle=False)

    # Updater

    epoch_count = 1
    niter = 100
    niter_decay = 100

    updater = CycleGANUpdater(
        iterator=iter_train,
        optimizer=dict(
            G_A=optimizer_G_A,
            G_B=optimizer_G_B,
            D_A=optimizer_D_A,
            D_B=optimizer_D_B,
        ),
        device=gpu,
    )

    # Trainer

    out = osp.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    out += suffix
    trainer = training.Trainer(
        updater, (niter + niter_decay, 'epoch'), out=out)

    trainer.extend(extensions.snapshot_object(
        target=G_A, filename='G_A_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=G_B, filename='G_B_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=D_A, filename='D_A_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        target=D_B, filename='D_B_{.updater.epoch:08}.npz'),
        trigger=(1, 'epoch'))

    trainer.extend(
        extensions.LogReport(trigger=(20 // batch_size, 'iteration')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_gen_A', 'loss_gen_B'],
        x_key='iteration', file_name='loss_gen.png',
        trigger=(100 // batch_size, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_dis_A', 'loss_dis_B'],
        x_key='iteration', file_name='loss_dis.png',
        trigger=(100 // batch_size, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_cyc_A', 'loss_cyc_B'],
        x_key='iteration', file_name='loss_cyc.png',
        trigger=(100 // batch_size, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['loss_idt_A', 'loss_idt_B'],
        x_key='iteration', file_name='loss_idt.png',
        trigger=(100 // batch_size, 'iteration')))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'elapsed_time',
        'loss_gen_A', 'loss_gen_B',
        'loss_dis_A', 'loss_dis_B',
        'loss_cyc_A', 'loss_cyc_B',
        'loss_idt_A', 'loss_idt_B',
    ]))

    trainer.extend(
        extensions.ProgressBar(update_interval=20 // batch_size))

    trainer.extend(CycleGANEvaluator(iter_test, device=gpu))

    @training.make_extension(trigger=(1, 'epoch'))
    def tune_learning_rate(trainer):
        epoch = trainer.updater.epoch

        lr_rate = 1.0 - (max(0, epoch + 1 + epoch_count - niter) /
                         float(niter_decay + 1))

        trainer.updater.get_optimizer('G_A').alpha *= lr_rate
        trainer.updater.get_optimizer('G_B').alpha *= lr_rate
        trainer.updater.get_optimizer('D_A').alpha *= lr_rate
        trainer.updater.get_optimizer('D_B').alpha *= lr_rate

    trainer.extend(tune_learning_rate)

    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True,
                        help='GPU id.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='Batch size.')
    args = parser.parse_args()

    dataset_train = TransformDataset(
        Horse2ZebraDataset('train'), CycleGANTransform())
    dataset_test = TransformDataset(
        Horse2ZebraDataset('test'), CycleGANTransform())
    train(dataset_train, dataset_test, args.gpu, args.batch_size)
