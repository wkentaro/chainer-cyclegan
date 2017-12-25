import chainer
from chainer import cuda
import chainer.functions as F
from chainer import training
import numpy as np


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
        })
