# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import os
import cv2
import time
import json
import copy
from utils.utils import get_category_list
import math

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, mode, cfg, root = './dataset/cifar', imb_type='exp',
                 transform=None, target_transform=None, download=True):
        train = True if mode == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.cfg = cfg
        self.train = train
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE

        rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            imb_factor = self.cfg.DATASET.IMBALANCECIFAR.RATIO
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.data_format_transform()
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

        self.data = self.all_info
        '''
            load the generated CAM-based dataset
        '''


        if self.cfg.DATASET.USE_CAM_BASED_DATASET and mode == 'train':
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = self.data + json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
            new_data = []
            for info in self.data:
                if 'fpath' not in info:
                    new_data.append(copy.deepcopy(info))
                    continue
                img = self._load_image(info)
                new_data.append({
                    'image': img,
                    'category_id': info['category_id']
                })
            self.data = new_data

        self.class_dict = self._get_class_dict()

        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
            print('-'*20+'in imbalance cifar dataset'+'-'*20)
            print('class_weight is: ')
            print(self.class_weight)

            num_list, cat_list = get_category_list(self.get_annotations(), self.cls_num, self.cfg)
            self.instance_p = np.array([num / sum(num_list) for num in num_list])
            self.class_p = np.array([1/self.cls_num for _ in num_list])
            num_list = [math.sqrt(num) for num in num_list]
            self.square_p = np.array([num / sum(num_list) for num in num_list])


    def update(self, epoch):
        self.epoch = max(0, epoch-self.cfg.TRAIN.TWO_STAGE.START_EPOCH) if self.cfg.TRAIN.TWO_STAGE.DRS else epoch
        if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train\
            and (not self.cfg.TRAIN.TWO_STAGE.DRS or (self.cfg.TRAIN.TWO_STAGE.DRS and self.epoch)):
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", 'square', 'progressive']
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
                sample_class = np.random.choice(np.arange(self.cls_num), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.cls_num), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index]['image'], self.data[index]['category_id']
        meta = dict()
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.cfg.TRAIN.SAMPLER.TYPE == "bbn sampler" and self.cfg.TRAIN.SAMPLER.BBN_SAMPLER.TYPE == "reverse":
            sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_img, sample_label = self.data[sample_index]['image'], self.data[sample_index]['category_id']
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)
            if self.target_transform is not None:
                sample_label = self.target_transform(sample_label)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label
        return img, target, meta

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print(fpath)
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _load_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight


    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        img = now_info['image']
        img = Image.fromarray(img)
        return self.transform(img)[None, :, :, :]

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for d in self.all_info:
            annos.append({'category_id': int(d['category_id'])})
        return annos

    def _get_image(self, now_info):
        img = now_info['image']
        return copy.deepcopy(img)

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            for img in self.data[selec_idx, ...]:
                new_data.append({
                    'image': img,
                    'category_id': the_class
                })
        self.all_info = new_data

    def data_format_transform(self):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        assert len(targets_np) == len(self.data)
        for i in range(len(self.data)):
            new_data.append({
                'image': self.data[i],
                'category_id': targets_np[i],
            })
        self.all_info = new_data


    def __len__(self):
        return len(self.all_info)



class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='/mnt/data3/zhangys/data/cifar', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)

