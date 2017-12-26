import os.path as osp
import subprocess

import chainer
import chainercv

from .base import UnpairedDirectoryDataset


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class BerkeleyCycleGANDataset(UnpairedDirectoryDataset):

    available_datasets = (
        'apple2orange',
        'summer2winter_yosemite',
        'horse2zebra',
        'monet2photo',
        'cezanne2photo',
        'ukiyoe2photo',
        'vangogh2photo',
        'maps',
        'cityscapes',
        'facades',
        'iphone2dslr_flower',
    )

    def __init__(self, name, split):
        if name not in self.available_datasets:
            raise ValueError('Unavailable dataset: {:s}'.format(name))
        self._name = name

        img_dir = osp.join(ROOT_DIR, self._name)
        if not osp.exists(img_dir):
            self.download()

        super(BerkeleyCycleGANDataset, self).__init__(img_dir, split)

    def download(self):
        url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{:s}.zip'.format(self._name)  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        # FIXME: *** Error in `python': munmap_chunk(): invalid pointer: 0x00007f315ba20990 ***  # NOQA
        # chainercv.utils.download.extractall(cache_path, ROOT_DIR, ext='.zip')
        subprocess.check_output(
            'unzip %s' % cache_path, cwd=ROOT_DIR, shell=True)


class Horse2ZebraDataset(BerkeleyCycleGANDataset):

    def __init__(self, split):
        super(Horse2ZebraDataset, self).__init__('horse2zebra', split=split)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = BerkeleyCycleGANDataset('horse2zebra', split='train')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        plt.subplot(121)
        plt.title('img_A')
        plt.imshow(img_A)
        plt.subplot(122)
        plt.title('img_B')
        plt.imshow(img_B)
        plt.show()
