import os
import os.path as osp
import shutil

import chainer
import chainercv
import numpy as np
import pandas

import skimage.io


ROOT_DIR = chainer.dataset.get_dataset_directory(
    'wkentaro/chainer-cyclegan/celebA')


# https://github.com/SKTBrain/DiscoGAN/blob/f0441787752c566dcc32642468f44d5e2f30b0a6/discogan/dataset.py#L56  # NOQA
def read_attr_file(attr_path, image_dir):
    f = open(attr_path)
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pandas.DataFrame(items, columns=columns)
    df['image_path'] = df['image_path'].map(lambda x: osp.join(image_dir, x))

    return df


def mkdir_p(*path):
    path = osp.join(*path)
    try:
        if not osp.exists(path):
            os.makedirs(path)
    except Exception:
        if not osp.isdir(path):
            raise
    return path


class CelebAStyle2StyleDataset(object):

    def __init__(self, split, style_A, style_B=None):
        assert split in ['train', 'test']
        self._split = split

        img_dir = osp.join(ROOT_DIR, 'Img/img_align_celeba')
        anno_file = osp.join(ROOT_DIR, 'Anno/list_attr_celeba.txt')

        if not osp.exists(img_dir) or not osp.exists(anno_file):
            self.download()

        df = read_attr_file(anno_file, img_dir)

        style_A_data = df[df[style_A] == '1']['image_path'].values
        if style_B is None:
            style_B_data = df[df[style_A] == '-1']['image_path'].values
        else:
            style_B_data = df[df[style_B] == '1']['image_path'].values

        n_test = 200

        if split == 'train':
            A, B = style_A_data[:-n_test], style_B_data[:-n_test]
        else:
            assert split == 'test'
            A, B = style_A_data[-n_test:], style_B_data[-n_test:]

        self._data = dict(A=A, B=B)
        self._size = {k: len(v) for k, v in self._data.items()}

    def __len__(self):
        return min(self._size.values())

    def __getitem__(self, index):
        path_A = self._data['A'][index % self._size['A']]
        if self._split == 'test':
            np.random.seed(index)
        path_B = np.random.choice(self._data['B'])

        img_A = skimage.io.imread(path_A)
        img_B = skimage.io.imread(path_B)

        return img_A, img_B

    @staticmethod
    def download():
        url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        dst_dir = mkdir_p(ROOT_DIR, 'Img')
        chainercv.utils.download.extractall(cache_path, dst_dir, ext='.zip')

        url = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1'  # NOQA
        cache_path = chainercv.utils.download.cached_download(url)
        dst_file = osp.join(mkdir_p(ROOT_DIR, 'Anno'), 'list_attr_celeba.txt')
        shutil.move(cache_path, dst_file)


if __name__ == '__main__':
    import cv2
    dataset = CelebAStyle2StyleDataset(split='train', style_A='Male')
    for i in range(len(dataset)):
        img_A, img_B = dataset[i]
        viz = np.hstack([img_A, img_B])
        cv2.imshow(__file__, viz[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break
