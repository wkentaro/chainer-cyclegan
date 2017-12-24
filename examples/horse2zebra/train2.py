#!/usr/bin/env python

from __future__ import print_function

import argparse
import copy
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'

import chainer
from chainer import cuda
from chainer.datasets import TransformDataset
import chainer.functions as F
import chainer.optimizers as O
from chainer import training
from chainer.training import extensions
import cupy as cp
import fcn
import numpy as np
import skimage.io
import skimage.util

from chainer_cyclegan.datasets import Horse2ZebraDataset
from chainer_cyclegan.datasets import transform
from chainer_cyclegan.models import NLayerDiscriminator
from chainer_cyclegan.models import ResnetGenerator


class ImagePool(object):

    def __init__(self, size):
        self._size = size
        self.pool = []

    def query(self, imgs):
        if self._size == 0:
            return imgs

        xp = cuda.get_array_module(imgs)

        res = []
        for img in imgs:
            img = img[None]
            if len(self.pool) < self._size:
                self.pool.append(img)
                res.append(img)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self._size)
                    tmp = self.pool[random_id].copy()
                    self.pool[random_id] = img
                    res.append(tmp)
                else:
                    res.append(img)
        res = chainer.Variable(xp.concatenate(res, axis=0))

        return res


class CycleGANUpdater(training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        super(CycleGANUpdater, self).__init__(*args, **kwargs)
        self._pool_fake_A = ImagePool(50)
        self._pool_fake_B = ImagePool(50)

    def loss_D(self, D, real, fake):
        xp = cuda.get_array_module(real.array)
        # real
        pred_real = D(real)
        loss_real = F.mean_squared_error(
            pred_real, xp.ones_like(pred_real.array))
        # fake
        pred_fake = D(fake.array)
        loss_fake = F.mean_squared_error(
            pred_fake, xp.zeros_like(pred_fake.array))
        # combined loss
        loss = (loss_real + loss_fake) * 0.5
        return loss

    def update_core(self):
        optimizer_G_A = self.get_optimizer('G_A')
        optimizer_D_A = self.get_optimizer('D_A')
        optimizer_G_B = self.get_optimizer('G_B')
        optimizer_D_B = self.get_optimizer('D_B')

        G_A = optimizer_G_A.target
        G_B = optimizer_G_B.target
        D_A = optimizer_D_A.target
        D_B = optimizer_D_B.target

        batch = next(self.get_iterator('main'))
        batch_A, batch_B = zip(*batch)
        real_A = chainer.Variable(self.converter(batch_A, self.device))
        real_B = chainer.Variable(self.converter(batch_B, self.device))

        xp = chainer.cuda.get_array_module(real_A.array)

        # parameters
        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0

        # update G
        # ---------------------------------------------------------------------
        G_A.zerograds()
        G_B.zerograds()

        # G_A should be identity if real_B is fed:
        # - b_i_hat = G_A(a_i)
        # - b_i = a_i_hat = G_A(b_i)
        idt_A = G_A(real_B)
        loss_idt_A = F.mean_absolute_error(idt_A, real_B) * \
            lambda_B * lambda_idt

        # G_B should be identity if real_A is fed.
        idt_B = G_B(real_A)
        loss_idt_B = F.mean_absolute_error(idt_B, real_A) * \
            lambda_A * lambda_idt

        # GAN loss D_A(G_A(A))
        fake_B = G_A(real_A)
        pred_fake = D_A(fake_B)
        loss_G_A = F.mean_squared_error(
            pred_fake, xp.ones_like(pred_fake.array))

        # GAN loss D_B(G_B(B))
        fake_A = G_B(real_B)
        pred_fake = D_B(fake_A)
        loss_G_B = F.mean_squared_error(
            pred_fake, xp.ones_like(pred_fake.array))

        # Forward cycle loss
        rec_A = G_B(fake_B)
        loss_cycle_A = F.mean_absolute_error(rec_A, real_A) * lambda_A

        # Backward cycle loss
        rec_B = G_A(fake_A)
        loss_cycle_B = F.mean_absolute_error(rec_B, real_B) * lambda_B

        # combined loss
        loss_G = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B +
                  loss_idt_A + loss_idt_B)
        loss_G.backward()

        optimizer_G_A.update()
        optimizer_G_B.update()

        # update D
        # ---------------------------------------------------------------------
        # backward D_A
        D_A.zerograds()
        fake_B = self._pool_fake_B.query(fake_B.array)
        loss_D_A = self.loss_D(D_A, real_B, fake_B)
        loss_D_A.backward()
        optimizer_D_A.update()

        # backward D_B
        D_B.zerograds()
        fake_A = self._pool_fake_A.query(fake_A.array)
        loss_D_B = self.loss_D(D_B, real_A, fake_A)
        loss_D_B.backward()
        optimizer_D_B.update()

        # report
        # ---------------------------------------------------------------------
        chainer.report({
            'loss_gen_A': loss_G_A,
            'loss_gen_B': loss_G_B,
            'loss_dis_A': loss_D_A,
            'loss_dis_B': loss_D_B,
            'loss_cyc_A': loss_cycle_A,
            'loss_cyc_B': loss_cycle_B,
            'loss_idt_A': loss_idt_A,
            'loss_idt_B': loss_idt_B,
        }, G_A)


class Evaluator(training.Extension):

    trigger = (1, 'epoch')

    def __init__(self, iterator,
                 converter=chainer.dataset.convert.concat_examples,
                 device=None,
                 shape=(3, 3)):
        self._iterator = iterator
        self.converter = converter
        self.device = device
        self._shape = shape

    def __call__(self, trainer):
        G_A = trainer.updater.get_optimizer('G_A').target
        G_B = trainer.updater.get_optimizer('G_B').target

        iterator = self._iterator
        it = copy.copy(iterator)

        vizs = []
        for batch in it:
            batch_A, batch_B = zip(*batch)
            real_A = chainer.Variable(self.converter(batch_A, self.device))
            real_B = chainer.Variable(self.converter(batch_B, self.device))

            idt_B = G_A(real_B)
            idt_A = G_B(real_A)

            fake_B = G_A(real_A)
            fake_A = G_B(real_B)

            rec_B = G_A(fake_A)
            rec_A = G_B(fake_B)

            real_A = cuda.to_cpu(real_A.array).transpose(0, 2, 3, 1)
            real_B = cuda.to_cpu(real_B.array).transpose(0, 2, 3, 1)
            idt_A = cuda.to_cpu(idt_A.array).transpose(0, 2, 3, 1)
            idt_B = cuda.to_cpu(idt_B.array).transpose(0, 2, 3, 1)
            fake_A = cuda.to_cpu(fake_A.array).transpose(0, 2, 3, 1)
            fake_B = cuda.to_cpu(fake_B.array).transpose(0, 2, 3, 1)
            rec_A = cuda.to_cpu(rec_A.array).transpose(0, 2, 3, 1)
            rec_B = cuda.to_cpu(rec_B.array).transpose(0, 2, 3, 1)

            real_A = ((real_A + 1) / 2 * 255).astype(np.uint8)
            real_B = ((real_B + 1) / 2 * 255).astype(np.uint8)
            idt_A = ((idt_A + 1) / 2 * 255).astype(np.uint8)
            idt_B = ((idt_B + 1) / 2 * 255).astype(np.uint8)
            fake_A = ((fake_A + 1) / 2 * 255).astype(np.uint8)
            fake_B = ((fake_B + 1) / 2 * 255).astype(np.uint8)
            rec_A = ((rec_A + 1) / 2 * 255).astype(np.uint8)
            rec_B = ((rec_B + 1) / 2 * 255).astype(np.uint8)

            batch_size = len(real_A)
            for i in range(batch_size):
                viz = np.vstack([
                    np.hstack([real_A[i], fake_B[i], rec_A[i], idt_A[i]]),
                    np.hstack([real_B[i], fake_A[i], rec_B[i], idt_B[i]]),
                ])
                vizs.append(viz)
                if len(vizs) >= (self._shape[0] * self._shape[1]):
                    break
            if len(vizs) >= (self._shape[0] * self._shape[1]):
                break

        out_file = osp.join(
            trainer.out, 'evaluations',
            '{:08}.png'.format(trainer.updater.epoch))
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        out = fcn.utils.get_tile_image(vizs, self._shape)
        skimage.io.imsave(out_file, out)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True)
    args = parser.parse_args()

    np.random.seed(0)
    if args.gpu >= 0:
        cp.random.seed(0)

    # Model

    G_A = ResnetGenerator()
    G_B = ResnetGenerator()
    D_A = NLayerDiscriminator()
    D_B = NLayerDiscriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
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
        TransformDataset(Horse2ZebraDataset('train'), transform),
        batch_size=1)
    iter_test = chainer.iterators.SerialIterator(
        TransformDataset(Horse2ZebraDataset('test'), transform),
        batch_size=1, repeat=False, shuffle=False)

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
        device=args.gpu,
    )

    # Trainer

    out = osp.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
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

    trainer.extend(extensions.LogReport(trigger=(20, 'iteration')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['G_A/loss_gen_A', 'G_A/loss_gen_B'],
        x_key='iteration', file_name='loss_gen.png',
        trigger=(100, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['G_A/loss_dis_A', 'G_A/loss_dis_B'],
        x_key='iteration', file_name='loss_dis.png',
        trigger=(100, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['G_A/loss_cyc_A', 'G_A/loss_cyc_B'],
        x_key='iteration', file_name='loss_cyc.png',
        trigger=(100, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['G_A/loss_idt_A', 'G_A/loss_idt_B'],
        x_key='iteration', file_name='loss_idt.png',
        trigger=(100, 'iteration')))

    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'elapsed_time',
        'G_A/loss_gen_A', 'G_A/loss_gen_B',
        'G_A/loss_dis_A', 'G_A/loss_dis_B',
        'G_A/loss_cyc_A', 'G_A/loss_cyc_B',
        'G_A/loss_idt_A', 'G_A/loss_idt_B',
    ]))

    trainer.extend(extensions.ProgressBar(update_interval=20))

    trainer.extend(Evaluator(iter_test, device=args.gpu))

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
    main()
