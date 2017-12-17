#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

# references
mkdir -p $ROOT/src
cd $ROOT/src
if [ ! -e chainer-cyclegan ]; then
  git clone https://github.com/Aixile/chainer-cyclegan.git chainer-cyclegan
fi
if [ ! -e pytorch-cyclegan ]; then
  git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pytorch-cyclegan
fi
cd -

# anaconda
if [ ! -e $ROOT/.anaconda2/bin/activate ]; then
  curl 'https://raw.githubusercontent.com/wkentaro/dotfiles/f6e3a7b23a5863676c66f740eec6c6ca7d7976fe/local/bin/install_anaconda2.sh' | bash -s $ROOT
fi
set +x && source $ROOT/.anaconda2/bin/activate && set -x
conda info -e

conda install pytorch cuda80 -c soumith -y
conda install opencv -c conda-forge -y
pip install chainer
