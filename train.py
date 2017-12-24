#!/usr/bin/env python

from __future__ import print_function

import argparse
import datetime
import os
import os.path as osp
import time

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.optimizers as O
import chainer.serializers as S
import cupy as cp
import numpy as np
import skimage.io

from chainer_cyclegan.datasets import Horse2ZebraDataset
from chainer_cyclegan.datasets import transform
from chainer_cyclegan.models import NLayerDiscriminator
from chainer_cyclegan.models import ResnetGenerator


class ImagePool(object):

    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        xp = cuda.get_array_module(images)

        return_images = []
        for image in images:
            image = image[None]
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].copy()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = chainer.Variable(xp.concatenate(return_images, axis=0))
        return return_images

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', type=int, required=True)
args = parser.parse_args()

gpu = args.gpu

G_A = ResnetGenerator()
G_B = ResnetGenerator()
D_A = NLayerDiscriminator()
D_B = NLayerDiscriminator()

if gpu >= 0:
    cuda.get_device_from_id(gpu).use()
    G_A.to_gpu()
    G_B.to_gpu()
    D_A.to_gpu()
    D_B.to_gpu()

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

optimizer_G_A = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
optimizer_G_A.setup(G_A)

optimizer_G_B = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
optimizer_G_B.setup(G_B)

optimizer_D_A = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
optimizer_D_A.setup(D_A)

optimizer_D_B = O.Adam(alpha=lr, beta1=beta1, beta2=beta2)
optimizer_D_B.setup(D_B)


batch_size = 1
dataset = Horse2ZebraDataset('train')

fake_A_pool = ImagePool(pool_size=50)
fake_B_pool = ImagePool(pool_size=50)


def backward_D_basic(D, real, fake):
    # Real
    pred_real = D(real)
    loss_D_real = F.mean_squared_error(
        pred_real, xp.ones_like(pred_real.array))
    # Fake
    pred_fake = D(fake.array)
    loss_D_fake = F.mean_squared_error(
        pred_fake, xp.zeros_like(pred_fake.array))
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    # backward
    loss_D.backward()
    return loss_D


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)  # NOQA
    return lr_l


epoch_count = 1
niter = 100
niter_decay = 100

out_dir = osp.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
if not osp.exists(out_dir):
    os.makedirs(out_dir)


with open(osp.join(out_dir, 'log.csv'), 'w') as f:
    f.write(','.join([
        'epoch',
        'iteration',
        'loss_G',
        'loss_G_A',
        'loss_G_B',
        'loss_idt_A',
        'loss_idt_B',
        'loss_cycle_A',
        'loss_cycle_B',
        'loss_D_A',
        'loss_D_B',
    ]))
    f.write('\n')


max_epoch = niter + niter_decay - epoch_count
dataset_size = len(dataset)
for epoch in range(epoch_count, niter + niter_decay + 1):
    t_start = time.time()

    for iteration in range(dataset_size):
        img_A, img_B = transform(dataset[iteration])

        assert batch_size == 1
        real_A = img_A[None, :, :, :]
        real_B = img_B[None, :, :, :]
        if gpu >= 0:
            real_A = cuda.to_gpu(real_A)
            real_B = cuda.to_gpu(real_B)
        real_A = chainer.Variable(real_A)
        real_B = chainer.Variable(real_B)

        # update G
        # -------------------------------------------------------------------------
        G_A.zerograds()
        G_B.zerograds()

        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0

        # G_A should be identity if real_B is fed:
        # - b_i_hat = G_A(a_i)
        # - b_i = a_i_hat = G_A(b_i)
        idt_A = G_A(real_B)
        loss_idt_A = F.mean_absolute_error(idt_A, real_B) * lambda_B * lambda_idt  # NOQA

        # G_B should be identity if real_A is fed.
        idt_B = G_B(real_A)
        loss_idt_B = F.mean_absolute_error(idt_B, real_A) * lambda_A * lambda_idt  # NOQA

        xp = cp if gpu >= 0 else np

        # GAN loss D_A(G_A(A))
        fake_B = G_A(real_A)
        pred_fake = D_A(fake_B)
        loss_G_A = F.mean_squared_error(pred_fake, xp.ones_like(pred_fake.array))  # NOQA

        # GAN loss D_B(G_B(B))
        fake_A = G_B(real_B)
        pred_fake = D_B(fake_A)
        loss_G_B = F.mean_squared_error(pred_fake, xp.ones_like(pred_fake.array))  # NOQA

        # Forward cycle loss
        rec_A = G_B(fake_B)
        loss_cycle_A = F.mean_absolute_error(rec_A, real_A) * lambda_A

        # Backward cycle loss
        rec_B = G_A(fake_A)
        loss_cycle_B = F.mean_absolute_error(rec_B, real_B) * lambda_B
        # combined loss
        loss_G = (loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B)  # NOQA
        loss_G.backward()

        optimizer_G_A.update()
        optimizer_G_B.update()

        # update D
        # ---------------------------------------------------------------------
        # backward D_A
        D_A.zerograds()
        fake_B_org = fake_B
        fake_B = fake_B_pool.query(fake_B.array)
        loss_D_A = backward_D_basic(D_A, real_B, fake_B)
        optimizer_D_A.update()

        # backward D_B
        D_B.zerograds()
        fake_A_org = fake_A
        fake_A = fake_A_pool.query(fake_A.array)
        loss_D_B = backward_D_basic(D_B, real_A, fake_A)
        optimizer_D_B.update()

        # log
        # ---------------------------------------------------------------------
        if iteration % 100 == 0:
            time_per_iter = (time.time() - t_start) / (iteration + 1)

            loss_G = float(loss_G.data)
            loss_G_A = float(loss_G_A.data)
            loss_G_B = float(loss_G_B.data)
            loss_D_A = float(loss_D_A.data)
            loss_D_B = float(loss_D_B.data)
            loss_cycle_A = float(loss_cycle_A.data)
            loss_cycle_B = float(loss_cycle_B.data)
            loss_idt_A = float(loss_idt_A.data)
            loss_idt_B = float(loss_idt_B.data)

            print('-' * 79)
            print('Epoch: {:d}/{:d} ({:.1%}), Iteration: {:d}/{:d} ({:.1%}), '
                  'Time: {:f}'
                  .format(epoch, max_epoch, 1. * epoch / max_epoch,
                          iteration, dataset_size,
                          1. * iteration / dataset_size, time_per_iter))

            print('G_A: {:.2f}'.format(loss_G_A),
                  'G_B: {:.2f}'.format(loss_G_B),
                  'D_A: {:.2f}'.format(loss_D_A),
                  'D_B: {:.2f}'.format(loss_D_B),
                  'C_A: {:.2f}'.format(loss_cycle_A),
                  'C_B: {:.2f}'.format(loss_cycle_B),
                  'I_A: {:.2f}'.format(loss_idt_A),
                  'I_B: {:.2f}'.format(loss_idt_B))

            with open(osp.join(out_dir, 'log.csv'), 'a') as f:
                f.write(','.join(map(str, [
                    epoch,
                    ((epoch - 1) * dataset_size) + iteration,
                    loss_G,
                    loss_G_A,
                    loss_G_B,
                    loss_idt_A,
                    loss_idt_B,
                    loss_cycle_A,
                    loss_cycle_B,
                    loss_D_A,
                    loss_D_B,
                ])))
                f.write('\n')

    # visualize
    # -------------------------------------------------------------------------
    real_A = real_A.array[0].transpose(1, 2, 0)
    real_B = real_B.array[0].transpose(1, 2, 0)
    real_A = cuda.to_cpu(real_A)
    real_B = cuda.to_cpu(real_B)
    fake_A = fake_A_org.array[0].transpose(1, 2, 0)
    fake_B = fake_B_org.array[0].transpose(1, 2, 0)
    fake_A = cuda.to_cpu(fake_A)
    fake_B = cuda.to_cpu(fake_B)
    rec_A = rec_A.array[0].transpose(1, 2, 0)
    rec_B = rec_B.array[0].transpose(1, 2, 0)
    rec_A = cuda.to_cpu(rec_A)
    rec_B = cuda.to_cpu(rec_B)
    viz = np.vstack([np.hstack([real_A, fake_B, rec_A]),
                     np.hstack([real_B, fake_A, rec_B])])
    skimage.io.imsave(osp.join(out_dir, '{:08}.jpg'.format(epoch)), viz)

    S.save_npz(osp.join(out_dir, '{:08}_G_A.npz'.format(epoch)), G_A)
    S.save_npz(osp.join(out_dir, '{:08}_G_B.npz'.format(epoch)), G_B)
    S.save_npz(osp.join(out_dir, '{:08}_D_A.npz'.format(epoch)), D_A)
    S.save_npz(osp.join(out_dir, '{:08}_D_B.npz'.format(epoch)), D_B)

    # update learning rate
    # -------------------------------------------------------------------------
    lr_new = lambda_rule(epoch)
    optimizer_G_A.alpha *= lr_new
    optimizer_G_B.alpha *= lr_new
    optimizer_D_A.alpha *= lr_new
    optimizer_D_B.alpha *= lr_new
