import chainer
from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import links
from chainer.utils import argument
from chainer import variable


class InstanceNormalization(links.BatchNormalization):

    def __call__(self, x, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x for instance normalization
        shape_org = x.shape
        B, C = shape_org[:2]
        shape_ins = (1, B * C) + shape_org[2:]
        x_reshaped = functions.reshape(x, shape_ins)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        gamma = functions.tile(gamma, (B,))
        beta = functions.tile(beta, (B,))
        mean = self.xp.tile(self.avg_mean, (B,))
        var = self.xp.tile(self.avg_var, (B,))

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.batch_normalization(
                x_reshaped, gamma, beta, eps=self.eps, running_mean=mean,
                running_var=var, decay=decay)

            self.avg_mean = mean.reshape(B, C).mean(axis=0)
            self.avg_var = var.reshape(B, C).mean(axis=0)
        else:
            # Use running average statistics or fine-tuned statistics.
            ret = functions.fixed_batch_normalization(
                x_reshaped, gamma, beta, mean, var, self.eps)

        # ret is normalized input x
        return functions.reshape(ret, shape_org)
