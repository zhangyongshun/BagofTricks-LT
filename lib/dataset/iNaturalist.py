from dataset.baseset import BaseSet
import random, cv2
import numpy as np


class iNaturalist(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super(iNaturalist, self).__init__(mode, cfg, transform)
        random.seed(0)
        self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" \
                and self.mode == 'train' \
                and (not self.cfg.TRAIN.TWO_STAGE.DRS or (self.cfg.TRAIN.TWO_STAGE.DRS and self.epoch)):
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", 'square', 'progressive']
            if  self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        meta = dict()
        image_label = now_info['category_id']  # 0-index
        return image, image_label, meta












