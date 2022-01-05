import numpy as np
import torch, math
from core.evaluate import accuracy
from net import Network

class Combiner:
    def __init__(self, cfg, device, num_class_list=None):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.num_class_list = torch.FloatTensor(num_class_list)
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()


    def initilize_all_parameters(self):
        self.alpha = self.cfg.TRAIN.COMBINER.ALPHA
        self.manifold_mix_up_location = self.cfg.TRAIN.COMBINER.MANIFOLD_MIX_UP.LOCATION
        self.remix_kappa = self.cfg.TRAIN.COMBINER.REMIX.KAPPA
        self.remix_tau = self.cfg.TRAIN.COMBINER.REMIX.TAU
        print('_'*100)
        print('combiner type: ', self.type)
        print('alpha in combiner: ', self.alpha)
        if self.type == 'manifold_mix_up':
            if 'res32' in self.cfg.BACKBONE.TYPE:
                assert self.manifold_mix_up_location in ['layer1', 'layer2', 'layer3', 'pool', 'fc']
            else:
                assert self.manifold_mix_up_location in ['layer1', 'layer2', 'layer3', 'pool', 'fc', 'layer4']
            print('location in manifold mixup: ', self.manifold_mix_up_location)
        if self.type == 'remix':
            print('kappa in remix: ', self.remix_kappa)
            print('tau in remix: ', self.remix_tau)
        print('_'*100)

    def update(self, epoch):
        self.epoch = epoch
    
    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def default(self, model, criterion, image, label, meta, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        if 'sample_image' in meta and 'sample_label' in meta:
            image_b = meta["sample_image"].to(self.device)
            label_b = meta["sample_label"].to(self.device)
            image = torch.cat([image, image_b], dim=0)
            label = torch.cat([label, label_b])

        feature = model(image, feature_flag=True)
        output = model(feature, classifier_flag=True, label=label)

        loss = criterion(output, label, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc

    def mix_up(self, model, criterion, image, label, meta, **kwargs):
        r"""
        References:
            Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR
        """
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]
        mixed_image = l * image_a + (1 - l) * image_b
        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)
        mixed_image = mixed_image.to(self.device)

        feature = model(mixed_image, feature_flag=True)
        output = model(feature, classifier_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0] + (1 - l) * \
                  accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]

        return loss, now_acc

    def manifold_mix_up(self, model, criterion, image, label, meta, **kwargs):
        r"""
        References:
            Verma et al., Manifold Mixup: Better Representations by Interpolating Hidden States, ICML 2019.

        Specially, we apply manifold mixup on only one layer in our experiments.
        The layer is assigned by param ``self.manifold_mix_up_location''
        """
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        label_a, label_b = label, label[idx]
        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)
        image = image.to(self.device)
        output = model(image, index=idx, layer=self.manifold_mix_up_location, coef=l)
        loss = l * criterion(output, label_a) + (1-l) * criterion(output, label_b)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0] + (1 - l) * \
                  accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        return loss, now_acc

    def remix(self, model, criterion, image, label, meta, **kwargs):
        r"""
        Reference:
            Chou et al. Remix: Rebalanced Mixup, ECCV 2020 workshop.

        The difference between input mixup and remix is that remix assigns lambdas of mixed labels
        according to the number of images of each class.

        Args:
            tau (float or double): a hyper-parameter
            kappa (float or double): a hyper-parameter
            See Equation (10) in original paper (https://arxiv.org/pdf/2007.03943.pdf) for more details.
        """
        assert self.num_class_list is not None, "num_class_list is required"

        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0))
        image_a, image_b = image, image[idx]
        label_a, label_b = label, label[idx]
        mixed_image = l * image_a + (1 - l) * image_b
        mixed_image = mixed_image.to(self.device)
        feature = model(mixed_image, feature_flag=True)
        output = model(feature, classifier_flag=True)

        #what remix does
        l_list = torch.empty(image.shape[0]).fill_(l).float().to(self.device)
        n_i, n_j = self.num_class_list[label_a], self.num_class_list[label_b].float()
        if l < self.remix_tau:
            l_list[n_i/n_j >= self.remix_kappa] = 0
        if 1 - l < self.remix_tau:
            l_list[(n_i*self.remix_kappa)/n_j <= 1] = 1

        label_a = label_a.to(self.device)
        label_b = label_b.to(self.device)
        loss = l_list * criterion(output, label_a) + (1 - l_list) * criterion(output, label_b)
        loss = loss.mean()
        now_result = torch.argmax(self.func(output), 1)
        now_acc = (l_list * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0] \
                + (1 - l_list) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]).mean()
        return loss, now_acc

    def bbn_mix(self, model, criterion, image, label, meta, **kwargs):
        r"""
        Reference:
            Zhou et al. BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition, CVPR 2020.

        We combine the sampling method of BBN, which consists of a uniform sampler and a reverse sampler, with input mixup.

        For more details about these two samplers, you can read the original paper https://arxiv.org/abs/1912.02413.
        """
        l = np.random.beta(self.alpha, self.alpha) # beta distribution

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        # mix up two image
        mixed_image = l * image_a + (1 - l) * image_b

        mixed_output = model(mixed_image)

        loss = l * criterion(mixed_output, label_a) + (1 - l) * criterion(mixed_output, label_b)

        now_result = torch.argmax(self.func(mixed_output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )
        return loss, now_acc

    def dive(self, model, criterion, image, label, meta, **kwargs):

        if not hasattr(self, 'model_t'):
            print('Loading the teacher model in DiVE')
            self.model_t = Network(self.cfg, mode="test", num_classes=len(self.num_class_list))
            self.model_t.load_model(self.cfg.TRAIN.COMBINER.DIVE.TEACHER_MODEL)
            self.model_t = torch.nn.DataParallel(self.model_t).cuda()
            self.model_t.eval()

        image, label = image.to(self.device), label.to(self.device)
        output_s = model(image)

        with torch.no_grad():
            output_t = self.model_t(image)

        loss = criterion(output_s, output_t, label)
        now_result = torch.argmax(self.func(output_s), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]
        return loss, now_acc