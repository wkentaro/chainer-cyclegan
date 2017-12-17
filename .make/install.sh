#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

mkdir -p $ROOT/src

cd $ROOT/src
if [ ! -e chainer-cyclegan ]; then
  git clone https://github.com/Aixile/chainer-cyclegan.git chainer-cyclegan
fi
if [ ! -e pytorch-cyclegan ]; then
  git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pytorch-cyclegan
fi
cd -
