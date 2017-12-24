#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL=https://people.eecs.berkeley.edu/~taesung_park/pytorch-CycleGAN-and-pix2pix/models/horse2zebra.pth
OUTPUT=$HERE/data/G_horse2zebra.pth

if [ -e $OUTPUT ]; then
  echo "File already exists: $OUTPUT"
else
  wget $URL -O $OUTPUT
fi
