## Bag of tricks for long-tailed visual recognition with deep convolutional neural networks

This repository is the official PyTorch implementation of [Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks](http://www.lamda.nju.edu.cn/zhangys/papers/AAAI_tricks.pdf), which provides practical and effective tricks used in long-tailed image classification. 



- #### Trick gallery:  ***[trick_gallery.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md)***. 
  - The tricks will be **constantly updated**.
- #### Recommond to install [Github Sort Content](https://github.com/Mottie/GitHub-userscripts/wiki/GitHub-sort-content), which can sort the columns of tables listed on github. With Github Sort Content, you can easily find the most efficient trick under each dataset in [trick_gallery.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md). You can see [this issue on stackoverflow](https://stackoverflow.com/questions/42843288/is-there-any-way-to-make-markdown-tables-sortable) for more information.


## Development log

- [x] `2021-05-19` - Add CONFIGs and experimental results of [BBN-style sampling](https://arxiv.org/abs/1912.02413) in [trick_gallery.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md), which consists of a uniform sampler and a reverse sampler.
- [x] `2021-05-15` - Add CONFIGs and experimental results of our bag of tricks in [trick_combination.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_combination.md).
<details><summary>Previous logs</summary>
<li> <strong>2021-04-24</strong> - Add the validation running command, which loads a trained model, then returns the validation acc and a corresponding confusion matrix figure. See <mark>Usage</mark> in this README for details.</li>
<li> <strong>2021-04-24</strong> - Add <a href="https://openreview.net/forum?id=r1gRTCVFvB">classifier-balancing</a> and corresponding experiments in Two-stage training in <a href="https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md">trick_gallery.md</a>, including $\tau$-normalization, cRT and LWS. </li>
<li> <strong>2021-04-23</strong> - Add CrossEntropyLabelAwareSmooth (<a href="https://arxiv.org/abs/2104.00466">label-aware smoothing, CVPR 2021</a>) in <a href="https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md">trick_gallery.md</a>. </li>
<li> <strong>2021-04-22</strong> - Add one option (TRAIN.APEX) in <a href="https://github.com/zhangyongshun/BagofTricks-LT/blob/main/lib/config/default.py">config.py</a>, so you can set TRAIN.APEX to False for training without using apex.</li>
<li> <strong>2021-01-30</strong> - Add the results of combining mixup methods and re-balancing in <a href="https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_combination.md">trick_combination.md</a>.</li>
<li> <strong>2021-02-19</strong> - Test and add the results of two-stage training in <a href="https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md">trick_gallery.md</a>.</li>
<li> <strong>2021-01-11</strong> - Add a mixup related method: <a href="https://arxiv.org/abs/2007.03943">Remix, ECCV 2020 workshop</a>.</li>
<li> <strong>2021-01-10</strong> - Add <a href="https://arxiv.org/abs/2001.01385">CDT (class-dependent temparature), arXiv 2020</a>, <a href="https://papers.nips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf">BSCE (balanced-softmax cross-entropy), NeurIPS 2020</a>, and support a smooth version of cost-sensitive cross-entropy (smooth CS_CE), which add a hyper-parameter $ \gamma$ to vanilla CS_CE. In smooth CS_CE, the loss weight of class i is defined as: $(\frac{N_{min}}{N_i})^\gamma$, where $\gamma \in [0, 1]$, $N_i$ is the number of images in class i. We can set $\gamma = 0.5$ to get a square-root version of CS_CE.</li>
<li> <strong>2021-01-05</strong> - Add <a href="https://arxiv.org/abs/2003.05176">SEQL (softmax equalization loss), CVPR 2020</a>.</li>
<li> <strong>2021-01-02</strong> - Add <a href="https://arxiv.org/abs/1906.07413">LDAMLoss, NeurIPS 2019</a>, and a regularization method: <a href="https://arxiv.org/abs/1512.00567">label smooth cross-entropy, CVPR 2016</a>.</li>
<li> <strong>2020-12-30</strong> - Add codes of torch.nn.parallel.DistributedDataParallel. Support apex in both torch.nn.DataParallel and torch.nn.parallel.DistributedDataParallel.</li>
</details>

## Trick gallery and combinations

#### Brief  inroduction

We divided the long-tail realted tricks into four families: re-weighting, re-sampling, mixup training, and two-stage training. For more details of the above four trick families, see the [original paper](https://cs.nju.edu.cn/wujx/paper/AAAI2021_Tricks.pdf).


#### Detailed information :

- Trick gallery:

  ##### Tricks, corresponding results, experimental settings, and running commands are listed in ***[trick_gallery.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_gallery.md)***.

- Trick combinations:

  ##### Combinations of different tricks, corresponding results, experimental settings, and running commands are listed in *[trick_combination.md](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/trick_combination.md)*.


## Main requirements
```bash
torch >= 1.4.0
torchvision >= 0.5.0
tensorboardX >= 2.1
tensorflow >= 1.14.0 #convert long-tailed cifar datasets from tfrecords to jpgs
Python 3
apex
```
- We provide the detailed requirements in [requirements.txt](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/documents/requirements.txt). You can run `pip install requirements.txt` to create the same running environment as ours.
- The [apex](https://github.com/NVIDIA/apex) **is recommended to be installed for saving GPU memories**:
```bash 
pip install -U pip
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- **If the apex is not installed, the `Distributed training with DistributedDataParallel` in our codes cannot be used.**
## Preparing the datasets

We provide three datasets in this repo: long-tailed CIFAR (CIFAR-LT), long-tailed ImageNet (ImageNet-LT), and iNaturalist 2018 (iNat18). 

The detailed information of these datasets are shown as follows:

<table>
<thead>
  <tr>
     <th align="center" rowspan="3">Datasets</th>
     <th align="center" colspan="2">CIFAR-10-LT</th>
     <th align="center" colspan="2">CIFAR-100-LT</th>
     <th align="center" rowspan="3">ImageNet-LT</th>
     <th align="center" rowspan="3">iNat18</th>
  </tr>
  <tr>
    <td align="center" colspan="4"><b>Imbalance factor</b></td>
  </tr>
  <tr>
     <td align="center" ><b>100</b></td>
     <td align="center" ><b>50</b></td>
     <td align="center" ><b>100</b></td>
     <td align="center" ><b>50</b></td>
  </tr>
</thead>
<tbody>
  <tr>
     <td align="center" style="font-weight:normal">    Training images</td>
     <td align="center" style="font-weight:normal">  12,406 </td>
     <td align="center" style="font-weight:normal">  13,996  </td>
     <td align="center" style="font-weight:normal">  10,847  </td>
     <td align="center" style="font-weight:normal"> 12,608 </td>
     <td align="center" style="font-weight:normal">11,5846</td>
     <td align="center" style="font-weight:normal">437,513</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">    Classes</td>
     <td align="center" style="font-weight:normal">  50  </td>
     <td align="center" style="font-weight:normal">   50 </td>
     <td align="center" style="font-weight:normal">   100 </td>
     <td align="center" style="font-weight:normal">  100  </td>
     <td align="center" style="font-weight:normal"> 1,000 </td>
     <td align="center" style="font-weight:normal">8,142</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">Max images</td>
     <td align="center" style="font-weight:normal">5,000</td>
     <td align="center" style="font-weight:normal">5,000</td>
     <td align="center" style="font-weight:normal">500</td>
     <td align="center" style="font-weight:normal">500</td>
     <td align="center" style="font-weight:normal">1,280</td>
     <td align="center" style="font-weight:normal">1,000</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal" >Min images</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">5</td>
     <td align="center" style="font-weight:normal">10</td>
     <td align="center" style="font-weight:normal">5</td>
     <td align="center" style="font-weight:normal">2</td>
  </tr>
  <tr>
     <td align="center" style="font-weight:normal">Imbalance factor</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">100</td>
     <td align="center" style="font-weight:normal">50</td>
     <td align="center" style="font-weight:normal">256</td>
     <td align="center" style="font-weight:normal">500</td>
  </tr>
</tbody>
</table>
<font size=2> -  `Max images` and `Min images` represents the number of training images in the largest and smallest classes, respectively.</font>

<font size=2> -  `CIFAR-10-LT-100` means the long-tailed CIFAR-10 dataset with the imbalance factor $\beta = 100$.</font>

<font size=2> -  `Imbalance factor` is defined as $\beta = \frac{\text{Max images}}{\text{Min images}}$.</font>

- #### Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/data/iNat18/images/train_val2018/Plantae/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```
- #### CIFAR-LT

  There are **two versions of CIFAR-LT**. 

  1. [Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555) firstly proposed the CIFAR-LT. They provided the [download link](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) of CIFAR-LT, and also the [codes](https://github.com/richardaecn/class-balanced-loss/blob/master/README.md#datasets) to generate the data, which are in TensorFlow. 

     You can follow the steps below to get this version of  CIFAR-LT:

     1. Download the Cui's CIFAR-LT in [GoogleDrive](https://drive.google.com/file/d/1NY3lWYRfsTWfsjFPxJUlPumy-WFeD7zK/edit) or [Baidu Netdisk ](https://pan.baidu.com/s/1rhTPUawY3Sky6obDM4Tczg) (password: 5rsq). Suppose you download the data and unzip them at path `/downloaded/data/`.
     2. Run tools/convert_from_tfrecords, and the converted CIFAR-LT and corresponding jsons will be generated at `/downloaded/converted/`.

  ```bash
  # Convert from the original format of CIFAR-LT
  python tools/convert_from_tfrecords.py  --input_path /downloaded/data/ --out_path /downloaded/converted/
  ```

  2. [Cao et al., NeurIPS 2019](https://arxiv.org/abs/1906.07413) followed  [Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555)'s method to generate the CIFAR-LT randomly. They modify the CIFAR datasets provided by PyTorch as [this file](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/lib/dataset/cao_cifar.py) shows.

- #### ImageNet-LT

  You can use the following steps to convert from the original images of ImageNet-LT.

  1. Download the original [ILSVRC-2012](http://www.image-net.org/). Suppose you have downloaded and reorgnized them at path `/downloaded/ImageNet/`, which should contain two sub-directories: `/downloaded/ImageNet/train` and `/downloaded/ImageNet/val`.
  2. Download the train/test splitting files (`ImageNet_LT_train.txt` and `ImageNet_LT_test.txt`) in [GoogleDrive](https://drive.google.com/drive/u/0/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-) or [Baidu Netdisk](https://pan.baidu.com/s/17alnFve8l-oMZFZQLxHrzQ) (password: cj0g).  Suppose you have downloaded them at path `/downloaded/ImageNet-LT/`. 
  3. Run tools/convert_from_ImageNet.py, and you will get two jsons: `ImageNet_LT_train.json` and `ImageNet_LT_val.json`.

  ```bash
  # Convert from the original format of ImageNet-LT
  python tools/convert_from_ImageNet.py --input_path /downloaded/ImageNet-LT/ --image_path /downloaed/ImageNet/ --output_path ./
  ```

- #### iNat18
  
  You can use the following steps to convert from the original format of iNaturalist 2018. 
  
  1. The images and annotations should be downloaded at [iNaturalist 2018](https://github.com/visipedia/inat_comp/blob/master/2018/README.md) firstly. Suppose you have downloaded them at  path `/downloaded/iNat18/`.
  2. Run tools/convert_from_iNat.py, and use the generated `iNat18_train.json` and `iNat18_val.json` to train.
  
  ```bash
  # Convert from the original format of iNaturalist
  # See tools/convert_from_iNat.py for more details of args 
  python tools/convert_from_iNat.py --input_json_file /downloaded/iNat18/train2018.json --image_path /downloaded/iNat18/images --output_json_file ./iNat18_train.json
  
  python tools/convert_from_iNat.py --input_json_file /downloaded/iNat18/val2018.json --image_path /downloaded/iNat18/images --output_json_file ./iNat18_val.json 
  ```

## Usage

In this repo:

- The results of CIFAR-LT (ResNet-32) and ImageNet-LT (ResNet-10), which need only one GPU to train, are gotten by DataParallel training with apex.

- The results of iNat18 (ResNet-50), which need more than one GPU to train, are gotten by DistributedDataParallel  training with apex. 

- If more than one GPU is used, DistributedDataParallel training is efficient than DataParallel training, especially when the CPU calculation forces are limited.

  
### Training
#### Parallel training with DataParallel 

```bash
1, To train
# To train long-tailed CIFAR-10 with imbalanced ratio of 50. 
# `GPUs` are the GPUs you want to use, such as `0,4`.
bash data_parallel_train.sh configs/test/data_parallel.yaml GPUs
```

#### Distributed training with DistributedDataParallel 
```bash
1, Change the NCCL_SOCKET_IFNAME in run_with_distributed_parallel.sh to [your own socket name]. 
export NCCL_SOCKET_IFNAME = [your own socket name]

2, To train
# To train long-tailed CIFAR-10 with imbalanced ratio of 50. 
# `GPUs` are the GPUs you want to use, such as `0,1,4`.
# `NUM_GPUs` are the number of GPUs you want to use. If you set `GPUs` to `0,1,4`, then `NUM_GPUs` should be `3`.
bash distributed_data_parallel_train.sh configs/test/distributed_data_parallel.yaml NUM_GPUs GPUs
```
### Validation

You can get the validation accuracy and the corresponding confusion matrix after running the following commands. 

See [main/valid.py](https://github.com/zhangyongshun/BagofTricks-LT/blob/main/main/valid.py) for more details.

```bash
1, Change the TEST.MODEL_FILE in the yaml to your own path of the trained model firstly.
2, To do validation
# `GPUs` are the GPUs you want to use, such as `0,1,4`.
python main/valid.py --cfg [Your yaml] --gpus GPUS
```
## The comparison between the baseline results using our codes and the references [<a href="https://arxiv.org/abs/1901.05555">Cui</a>, <a href="https://arxiv.org/abs/1910.09217">Kang</a>]

- We use **Top-1 error rates** as our evaluation metric.

<!--ImageNet_LT baseline acc, baseline_noc 35.01 baseline 36.26 baseline2 34.18 baseline2_noc 33.71 -->

- From the results of two CIFAR-LT, we can see that the **CIFAR-LT provided by [Cao](https://arxiv.org/abs/1906.07413) has much lower Top-1 error rates on CIFAR-10-LT**, compared with the baseline results reported in his paper. So, **in our experiments, we use the CIFAR-LT of [Cui](https://arxiv.org/abs/1901.05555) for fairness.**
- **For the ImageNet-LT, we find that  the color_jitter augmentation was not included in our experiments, which, however, is adopted by other methods. So, in this repo, we add the color_jitter augmentation on ImageNet-LT. The old baseline without color_jitter is 64.89, which is +1.15 points higher than the new baseline.**
- You can click the `Baseline` in the table below to see the experimental settings and corresponding running commands. 

<table>
<thead>
  <tr>
    <th rowspan="4" align="center">Datasets</th>
    <th align="center" colspan="4" align="center"><a href="https://arxiv.org/abs/1901.05555">Cui et al., 2019</a></th>
    <th align="center" colspan="4"><a href="https://arxiv.org/abs/1906.07413">Cao et al., 2020</a></th>
    <th align="center" rowspan="4">ImageNet-LT</th>
    <th align="center" rowspan="4">iNat18</th>
  </tr>
  <tr>
    <th align="center"  colspan="2">CIFAR-10-LT</td>
    <th align="center"  colspan="2">CIFAR-100-LT</td>
    <th align="center"  colspan="2">CIFAR-10-LT</td>
    <th align="center"  colspan="2">CIFAR-100-LT</td>
  </tr>
  <tr>
    <th align="center"  colspan="4" align="center">Imbalance factor</td>
    <th align="center"  colspan="4">Imbalance factor</td>
  </tr>
  <tr>
    <th align="center" >100</td>
    <th align="center" >50</td>
    <th align="center" >100</td>
    <th align="center" >50</td>
    <th align="center" >100</td>
    <th align="center" >50</td>
    <th align="center" >100</td>
    <th align="center" >50</td>
  </tr>
</thead>
<tbody>
  <tr>
    <th align="center" style="font-weight:normal">Backbones</td>
    <th align="center"  colspan="4" style="font-weight:normal">ResNet-32</td>
    <th align="center"  colspan="4" style="font-weight:normal">ResNet-32</td>
    <th align="center" style="font-weight:normal">ResNet-10</td>
    <th align="center" style="font-weight:normal" >ResNet-50</td>
  </tr>
  <tr>
    <th align="left" style="font-weight:normal"><details><summary>Baselines using our codes</summary>
      <ol>
      <li>CONFIG (from left to right): 
        <ul>
          <li>configs/cui_cifar/baseline/{cifar10_im100.yaml, cifar10_im50.yaml, cifar100_im100.yaml, cifar100_im50.yaml}</li> 
          <li>configs/cao_cifar/baseline/{cifar10_im100.yaml, cifar10_im50.yaml, cifar100_im100.yaml, cifar100_im50.yaml}</li>
          <li>configs/ImageNet_LT/imagenetlt_baseline.yaml</li>
          <li>configs/iNat18/iNat18_baseline.yaml</li>
        </ul>
        </li><br/>
      <li>Running commands:
        <ul>
          <li>For CIFAR-LT and ImageNet-LT: bash data_parallel_train.sh CONFIG GPU</li>
          <li>For iNat18: bash distributed_data_parallel_train.sh configs/iNat18/iNat18_baseline.yaml NUM_GPUs GPUs</li>
        </ul>
      </li>
      </ol>
      </details></td>
    <th align="center" style="font-weight:normal">30.12</td>
    <th align="center" style="font-weight:normal">24.81</td>
    <th align="center" style="font-weight:normal">61.76</td>
    <th align="center" style="font-weight:normal">57.65</td>
    <th align="center" style="font-weight:normal">28.05</td>
    <th align="center" style="font-weight:normal">23.55</td>
    <th align="center" style="font-weight:normal">62.27</td>
    <th align="center" style="font-weight:normal">56.22</td>
    <th align="center" style="font-weight:normal">63.74</td>
    <th align="center" style="font-weight:normal">40.55</td>
  </tr>
  <tr>
    <th align="left" style="font-weight:normal">Reference [<a href="https://arxiv.org/abs/1901.05555">Cui</a>, <a href="https://arxiv.org/abs/1910.09217">Kang</a>, <a href="https://arxiv.org/abs/1904.05160">Liu</a>]</td>
    <th align="center" style="font-weight:normal">29.64</td>
    <th align="center" style="font-weight:normal">25.19</td>
    <th align="center" style="font-weight:normal">61.68</td>
    <th align="center" style="font-weight:normal">56.15</td>
    <th align="center" style="font-weight:normal">29.64</td>
    <th align="center" style="font-weight:normal">25.19</td>
    <th align="center" style="font-weight:normal">61.68</td>
    <th align="center" style="font-weight:normal">56.15</td>
    <th align="center" style="font-weight:normal">64.40</td>
    <th align="center" style="font-weight:normal">42.86</td>
  </tr>
</tbody>
</table>

## Paper collection of long-tailed visual recognition

[Awesome-of-Long-Tailed-Recognition](https://github.com/zwzhang121/Awesome-of-Long-Tailed-Recognition)

[Long-Tailed-Classification-Leaderboard](https://github.com/yanyanSann/Long-Tailed-Classification-Leaderboard)

## Citation

```
@inproceedings{zhang2020tricks,
  author    = {Yongshun Zhang and Xiu{-}Shen Wei and Boyan Zhou and Jianxin Wu},
  title     = {Bag of Tricks for Long-Tailed Visual Recognition with Deep Convolutional Neural Networks},
  pages = {3447--3455},
  booktitle = {AAAI},
  year      = {2021},
}
```


## Contacts

If you have any question about our work, please do not hesitate to contact us by emails provided in the [paper](http://www.lamda.nju.edu.cn/zhangys/papers/AAAI_tricks.pdf).