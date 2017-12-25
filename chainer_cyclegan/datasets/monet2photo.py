import os.path as osp

import chainer
import chainercv

from .base import UnpairedDatasetBase


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class Monet2PhotoDataset(UnpairedDatasetBase):

    def __init__(self, split):
        img_dir = osp.join(ROOT_DIR, 'monet2photo')
        if not osp.exists(img_dir):
            self.download()
        super(Monet2PhotoDataset, self).__init__(img_dir, split)

    def download(self):
        url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        chainercv.utils.download.extractall(cache_path, ROOT_DIR, ext='.zip')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = Monet2PhotoDataset(split='train')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        plt.subplot(121)
        plt.title('img_A')
        plt.imshow(img_A)
        plt.subplot(122)
        plt.title('img_B')
        plt.imshow(img_B)
        plt.show()
