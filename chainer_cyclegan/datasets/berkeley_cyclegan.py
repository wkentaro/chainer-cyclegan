import os.path as osp

import chainer
import chainercv

from .base import UnpairedDatasetBase


ROOT_DIR = chainer.dataset.get_dataset_directory('wkentaro/chainer-cyclegan')


class BerkeleyCycleGANDataset(UnpairedDatasetBase):

    def __init__(self, name, split):
        if name not in self.available_datasets:
            raise ValueError('Unavailable dataset: {:s}'.format(name))

        img_dir = osp.join(ROOT_DIR, name)
        if not osp.exists(img_dir):
            self.download()

        super(BerkeleyCycleGANDataset, self).__init__(img_dir, split)

    def download(self):
        url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{:s}.zip'.format(name)  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        chainercv.utils.download.extractall(cache_path, ROOT_DIR, ext='.zip')

    @property
    def available_datasets(self):
        return (
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
            'ae_photos',
        )


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
