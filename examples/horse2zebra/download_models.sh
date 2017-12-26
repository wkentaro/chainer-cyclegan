#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x
set -e

# ----------------------------------------------------------------------------------------------------------

URL=https://people.eecs.berkeley.edu/~taesung_park/pytorch-CycleGAN-and-pix2pix/models/horse2zebra.pth
OUTPUT=$HERE/data/G_horse2zebra.pth

if [ -e $OUTPUT ]; then
  echo "File already exists: $OUTPUT"
else
  wget $URL -O $OUTPUT
fi

# ----------------------------------------------------------------------------------------------------------

URL=https://drive.google.com/uc?id=1j3D4UrKUN5RZW5qDQgy7tJnWtnQeV-Tc
OUTPUT=$HERE/data/G_horse2zebra_from_pytorch.npz

if [ -e $OUTPUT ]; then
  echo "File already exists: $OUTPUT"
else
  which gdown &>/dev/null || pip install gdown
  gdown $URL -O $OUTPUT
fi
