#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
OUTPUT=$HERE/data/horse2zebra.zip

if [ -e $OUTPUT ]; then
  echo "File already exists: $OUTPUT"
else
  wget $URL -O $OUTPUT
  unzip $OUTPUT -d $(dirname $OUTPUT)
fi
