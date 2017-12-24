#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

# references
mkdir -p $ROOT/src
cd $ROOT/src
if [ ! -e chainer-cyclegan ]; then
  git clone https://github.com/Aixile/chainer-cyclegan.git chainer-cyclegan -b 11454ae00fd6cde972d64673086ca3cded98f504
fi
if [ ! -e pytorch-cyclegan ]; then
  git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git pytorch-cyclegan -b 929454c133fb19c03264b00179dae25f458efd36
fi
cd -

# anaconda
if [ ! -e $ROOT/.anaconda2/bin/activate ]; then
  curl 'https://raw.githubusercontent.com/wkentaro/dotfiles/f6e3a7b23a5863676c66f740eec6c6ca7d7976fe/local/bin/install_anaconda2.sh' | bash -s $ROOT
fi
set +x && source $ROOT/.anaconda2/bin/activate && set -x
conda info -e

python -c 'import torch' &>/dev/null || conda install pytorch cuda80 -c soumith -y
python -c 'import cv2' &>/dev/null || conda install opencv -c menpo -y
python -c 'import chainer' &>/dev/null || pip install chainer
python -c 'import skimage' &>/dev/null || pip install scikit-image

# dev
python -c 'import IPython' &>/dev/null || pip install ipython
python -c 'import pipdeptree' &>/dev/null || pip install pipdeptree
python -c 'import pandas' &>/dev/null || pip install pandas
python -c 'import seaborn' &>/dev/null || pip install seaborn
