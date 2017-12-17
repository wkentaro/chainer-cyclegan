#!/bin/bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -x
set -e

set +x && source $ROOT/.anaconda2/bin/activate && set -x

cd $ROOT
python infer_pytorch.py # data/G_horse2zebra.pth
python infer_chainer.py # data/G_horse2zebra_from_pytorch.npz
