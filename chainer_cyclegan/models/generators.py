import chainer
import chainer.functions as F
import chainer.links as L

from chainer_cyclegan.links import InstanceNormalization


class ResnetBlock(chainer.Chain):

    def __init__(self):
        super(ResnetBlock, self).__init__()

        initialW = chainer.initializers.Normal(scale=0.02)
        with self.init_scope():
            self.l0 = lambda x: F.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)],
                                      mode='reflect')
            self.l1 = L.Convolution2D(256, 256, ksize=3, stride=1,
                                      initialW=initialW)
            self.l2 = InstanceNormalization(256, decay=0.9, eps=1e-05)
            self.l3 = lambda x: F.relu(x)
            self.l4 = lambda x: F.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1)],
                                      mode='reflect')
            self.l5 = L.Convolution2D(256, 256, ksize=3, stride=1,
                                      initialW=initialW)
            self.l6 = InstanceNormalization(256, decay=0.9, eps=1e-05)

        self.functions = []
        for i in range(0, 7):
            self.functions.append(getattr(self, 'l{:d}'.format(i)))

    def __call__(self, x):
        h = x
        for i, func in enumerate(self.functions):
            h = func(h)
        h = x + h
        return h


class ResnetGenerator(chainer.Chain):

    def __init__(self):
        super(ResnetGenerator, self).__init__()

        initialW = chainer.initializers.Normal(scale=0.02)
        with self.init_scope():
            self.l0 = lambda x: F.pad(x, [(0, 0), (0, 0), (3, 3), (3, 3)],
                                      mode='reflect')
            self.l1 = L.Convolution2D(3, 64, ksize=7, stride=1,
                                      initialW=initialW)
            # Chainer <-> PyTorch
            # * decay=0.9 <-> momentum=0.1
            #   (FIXME: https://github.com/keras-team/keras/issues/6839)
            # * use_gamma=False, use_beta=False <-> affine=False
            self.l2 = InstanceNormalization(64, decay=0.9, eps=1e-05)
            self.l3 = lambda x: F.relu(x)
            self.l4 = L.Convolution2D(64, 128, ksize=3, stride=2, pad=1,
                                      initialW=initialW)
            self.l5 = InstanceNormalization(128, decay=0.9, eps=1e-05)
            self.l6 = lambda x: F.relu(x)
            self.l7 = L.Convolution2D(128, 256, ksize=3, stride=2, pad=1,
                                      initialW=initialW)
            self.l8 = InstanceNormalization(256, decay=0.9, eps=1e-05)
            self.l9 = lambda x: F.relu(x)
            self.l10 = ResnetBlock()
            self.l11 = ResnetBlock()
            self.l12 = ResnetBlock()
            self.l13 = ResnetBlock()
            self.l14 = ResnetBlock()
            self.l15 = ResnetBlock()
            self.l16 = ResnetBlock()
            self.l17 = ResnetBlock()
            self.l18 = ResnetBlock()
            self.l19 = L.Deconvolution2D(256, 128, ksize=3, stride=2, pad=1,
                                         initialW=initialW)
            self.l20 = InstanceNormalization(128, decay=0.9, eps=1e-05)
            self.l21 = lambda x: F.relu(x)
            self.l22 = L.Deconvolution2D(128, 64, ksize=3, stride=2, pad=1,
                                         initialW=initialW)
            self.l23 = InstanceNormalization(64, decay=0.9, eps=1e-05)
            self.l24 = lambda x: F.relu(x)
            self.l25 = lambda x: F.pad(x, [(0, 0), (0, 0), (3, 3), (3, 3)],
                                       mode='reflect')
            self.l26 = L.Convolution2D(64, 3, ksize=7, stride=1,
                                       initialW=initialW)
            self.l27 = lambda x: F.tanh(x)

        self.functions = []
        for i in range(0, 28):
            self.functions.append(getattr(self, 'l{:d}'.format(i)))

    def __call__(self, x):
        h = x
        for i, func in enumerate(self.functions):
            if isinstance(func, L.Deconvolution2D):
                # 1 padding to the output
                outsize_h = chainer.utils.conv.get_deconv_outsize(
                    size=h.shape[2], k=func.ksize, s=func.stride[0],
                    p=func.pad[0])
                outsize_w = chainer.utils.conv.get_deconv_outsize(
                    size=h.shape[3], k=func.ksize, s=func.stride[1],
                    p=func.pad[1])
                func.outsize = (outsize_h + 1, outsize_w + 1)
                h = func(h)
            else:
                h = func(h)
        return h
