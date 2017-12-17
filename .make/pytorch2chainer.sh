#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

set +x && source $ROOT/.anaconda2/bin/activate && set -x

# pytorch-cyclegan
if [ ! -e $ROOT/data/G_horse2zebra.pth ]; then
  cd $ROOT/src/pytorch-cyclegan/pretrained_models
  bash ./download_cyclegan_model.sh horse2zebra
  mv $ROOT/src/pytorch-cyclegan/pretrained_models/checkpoints/horse2zebra_pretrained/latest_net_G.pth $ROOT/data/G_horse2zebra.pth
  cd -
fi

cd $ROOT
if [ ! -e $ROOT/data/G_horse2zebra_from_pytorch.npz ]; then
  python pytorch2chainer_G.py
fi
if [ ! -e $ROOT/data/D_random_from_pytorch.npz ]; then
  python generate_D_random_pth.py
  python pytorch2chainer_D.py
fi
