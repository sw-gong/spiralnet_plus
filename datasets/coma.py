import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh

from tqdm import tqdm


class CoMA(InMemoryDataset):
    url = 'https://coma.is.tue.mpg.de/'

    categories = [
        'bareteeth',
        'cheeks_in',
        'eyebrow',
        'high_smile',
        'lips_back',
        'lips_up',
        'mouth_down',
        'mouth_extreme',
        'mouth_middle',
        'mouth_open',
        'mouth_side',
        'mouth_up',
    ]

    def __init__(self,
                 root,
                 train=True,
                 split='interpolation',
                 test_exp='bareteeth',
                 transform=None,
                 pre_transform=None):
        self.split = split
        self.test_exp = test_exp
        if not osp.exists(osp.join(root, 'processed', self.split)):
            os.makedirs(osp.join(root, 'processed', self.split))
        if self.split == 'extrapolation':
            if self.test_exp not in self.categories:
                raise RuntimeError(
                    'Expected expressions in {}, but found {}'.format(
                        self.categories, self.test_exp))
            if not osp.exists(
                    osp.join(root, 'processed', self.split, self.test_exp)):
                os.makedirs(
                    osp.join(root, 'processed', self.split, self.test_exp))
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'COMA_data.zip'

    @property
    def processed_file_names(self):
        if self.split == 'extrapolation':
            return [
                osp.join(self.split, self.test_exp, 'training.pt'),
                osp.join(self.split, self.test_exp, 'test.pt')
            ]
        elif self.split == 'interpolation':
            return [
                osp.join(self.split, 'training.pt'),
                osp.join(self.split, 'test.pt')
            ]
        else:
            raise RuntimeError(
                ('Expected the split of interpolation or extrapolation, but'
                 ' found {}').format(self.split))

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download COMA_data.zip from {} and '
            'move it to {}'.format(self.url, self.raw_dir))

    def process(self):
        print('Processing...')
        fps = glob(osp.join(self.raw_dir, '*/*/*.ply'))
        if len(fps) == 0:
            extract_zip(self.raw_paths[0], self.raw_dir, log=False)
            fps = glob(osp.join(self.raw_dir, '*/*/*.ply'))

        train_data_list, test_data_list = [], []
        for idx, fp in enumerate(tqdm(fps)):
            data = read_mesh(fp)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if self.split == 'interpolation':
                if (idx % 100) < 10:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)
            elif self.split == 'extrapolation':
                if fp.split('/')[-2] == self.test_exp:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)
            else:
                raise RuntimeError((
                    'Expected the split of interpolation or extrapolation, but'
                    ' found {}').format(self.split))

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
