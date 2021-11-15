from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2
import torchvision.transforms as transforms
from data_transform.transform_wrapper import TRANSFORMS
import numpy as np
from utils.utils import get_category_list
import math
from PIL import Image

class BaseSet(Dataset):
    def __init__(self, mode="train", cfg=None, transform=None):
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))


        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = cfg.DATASET.TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET.VALID_JSON
        else:
            raise NotImplementedError
        self.update_transform()

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]

        if not self.cfg.DATASET.USE_CAM_BASED_DATASET or self.mode != 'train':
            self.data = self.all_info['annotations']
        else:
            assert os.path.isfile(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH), \
                'the CAM-based generated json file does not exist!'
            self.data = self.all_info['annotations'] + json.load(open(self.cfg.DATASET.CAM_DATA_JSON_SAVE_PATH))
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))
        self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)

        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and mode == "train":
            num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

            self.instance_p = np.array([num / sum(num_list) for num in num_list])
            self.class_p = np.array([1/self.num_classes for _ in num_list])
            num_list = [math.sqrt(num) for num in num_list]
            self.square_p = np.array([num / sum(num_list) for num in num_list])
            self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = max(0, epoch-self.cfg.TRAIN.TWO_STAGE.START_EPOCH) if self.cfg.TRAIN.TWO_STAGE.DRS else epoch
        if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)


    def __getitem__(self, index):
        print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
        )  # 0-index
        if self.mode not in ["train", "valid"]:
           meta["image_id"] = now_info["image_id"]
           meta["fpath"] = now_info["fpath"]

        return image, image_label, meta

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.all_info['annotations']

    def __len__(self):
        return len(self.all_info['annotations'])

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

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)[None, :, :, :]

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
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
        class_weight = [max_num / i if i != 0 else 0 for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

