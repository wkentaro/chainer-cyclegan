all:

install:
	.make/install.sh

pytorch2chainer: install
	.make/pytorch2chainer.sh

test_pytorch2chainer: pytorch2chainer
	.make/test_pytorch2chainer.sh
