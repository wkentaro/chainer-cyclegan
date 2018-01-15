import copy
import os
import os.path as osp

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import training
import fcn
import numpy as np
import skimage.io


class CycleGANEvaluator(training.Extension):

    trigger = (1, 'epoch')

    def __init__(self, iterator,
                 converter=convert.concat_examples,
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

            with chainer.using_config('enable_backprop', False), \
                    chainer.using_config('train', False):
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
                viz = fcn.utils.get_tile_image([
                    real_A[i], fake_B[i], rec_A[i], idt_A[i],
                    real_B[i], fake_A[i], rec_B[i], idt_B[i],
                ], tile_shape=(2, 4))
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
