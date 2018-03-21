import os.path as osp

import chainer
import chainercv

from .berkeley_cyclegan import ROOT_DIR
from .paired import PairedDirectoryDataset


class BerkeleyPix2PixDataset(PairedDirectoryDataset):

    available_datasets = (
        'edges2handbags',
        'edges2shoes',
    )

    def __init__(self, name, split):
        if name not in self.available_datasets:
            raise ValueError('Unavailable dataset: {:s}'.format(name))
        self._name = name

        img_dir = osp.join(ROOT_DIR, name)
        if not osp.exists(img_dir):
            self.download()

        super(BerkeleyPix2PixDataset, self).__init__(img_dir, split)

    def download(self):
        url = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/{:s}.tar.gz'.format(self._name)  # NOQA
        cache_path = chainer.dataset.cached_download(url)
        chainercv.utils.extractall(cache_path, ROOT_DIR, ext='.tgz')


if __name__ == '__main__':
    import cv2
    import numpy as np
    # dataset = BerkeleyPix2PixDataset(name='edges2handbags', split='train')
    dataset = BerkeleyPix2PixDataset(name='edges2shoes', split='train')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        viz = np.hstack([img_A, img_B])
        cv2.imshow(__file__, viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break
