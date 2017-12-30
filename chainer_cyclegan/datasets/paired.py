import glob
import os.path as osp

import cv2


def _imread_as_rgb(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1]  # BGR -> RGB


class PairedDatasetBase(object):

    def __init__(self, split, paths):
        assert split in ['train', 'val']
        self._split = split
        self._paths = paths
        self._size = len(self._paths)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        path = self._paths[index]
        img_AB = _imread_as_rgb(path)

        W = img_AB.shape[1]
        assert W % 2 == 0
        W_half = W // 2

        img_A = img_AB[:, :W_half, :]
        img_B = img_AB[:, W_half:, :]

        return img_A, img_B


class PairedDirectoryDataset(PairedDatasetBase):

    def __init__(self, img_dir, split):
        paths = []
        split_dir = osp.join(img_dir, split)
        for img_file in glob.glob(osp.join(split_dir, '*')):
            paths.append(img_file)

        super(PairedDirectoryDataset, self).__init__(split=split, paths=paths)
