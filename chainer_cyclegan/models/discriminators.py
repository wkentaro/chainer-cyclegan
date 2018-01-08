import chainer
import chainer.functions as F
import chainer.links as L

from chainer_cyclegan.links import InstanceNormalization


class NLayerDiscriminator(chainer.Chain):

    def __init__(self, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        initialW = chainer.initializers.Normal(scale=0.02)
        with self.init_scope():
            functions = [
                L.Convolution2D(3, 64, ksize=4, stride=2, pad=1,
                                initialW=initialW),
                lambda x: F.leaky_relu(x, 0.2),
            ]

            n_layers = 3

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                functions += [
                    L.Convolution2D(ndf * nf_mult_prev, ndf * nf_mult,
                                    ksize=4, stride=1, pad=1),
                    InstanceNormalization(ndf * nf_mult, decay=0.9, eps=1e-5),
                    lambda x: F.leaky_relu(x, 0.2),
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            functions += [
                L.Convolution2D(ndf * nf_mult_prev, ndf * nf_mult,
                                ksize=4, stride=1, pad=1),
                InstanceNormalization(ndf * nf_mult, decay=0.9, eps=1e-5),
                lambda x: F.leaky_relu(x, 0.2),
            ]

            functions += [
                L.Convolution2D(ndf * nf_mult, 1, ksize=4, stride=1, pad=1),
            ]

            for i, func in enumerate(functions):
                setattr(self, 'l{:d}'.format(i), func)

        self.functions = functions

    def __call__(self, x):
        h = x
        for i, func in enumerate(self.functions):
            h = func(h)
        return h
