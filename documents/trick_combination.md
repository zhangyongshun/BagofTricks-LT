# Trick combinations, corresponding results, experimental settings, and running commands

### 1. Mixup methods with re-balancing 

- Corresponding CONFIGs at [this link]().
- In this section, **we will show that the mixup shows excellent performance, when combined with re-balancing.** We choose the following methods to conduct experiments:
  - Mixup methods: input mixup, manifold mixup, and remix.
  - Re-balancing methods: re-weighting (CS_CE, BalancedSoftmaxCE) and re-sampling (Class-balanced sampling, Progressively-balanced sampling).

|      Mixup      |     Re-weighting      |           Re-sampling           | CIFAR-10-LT-100 | CIFAR-10-LT-50 | CIFAR-100-LT-100 | CIFAR-100-LT-50 |
| :-------------: | :-------------------: | :-----------------------------: | :-------------: | :------------: | :--------------: | :-------------: |
|       --        |         CE_CE         |               --                |      30.12      |     24.81      |      61.76       |      57.65      |
|       --        |   BalancedSoftmaxCE   |               --                |      23.63      |     20.07      |      56.82       |      54.55      |
|       --        |          --           |     Class-balanced sampling     |      30.35      |     23.38      |      66.47       |      61.07      |
|       --        |          --           | Progressively-balanced sampling |      29.35      |     24.06      |      60.90       |      56.92      |
|   Input mixup   |          --           |               --                |      28.55      |     21.83      |      60.10       |      55.25      |
| Manifold mixup  |          --           |               --                |      27.88      |     21.68      |      60.53       |      56.76      |
|      Remix      |          --           |               --                |      29.50      |     23.44      |      59.13       |      54.62      |
|   Input mixup   |         CS_CE         |               --                |      23.31      |     18.86      |      61.40       |      55.54      |
| **Input mixup** | **BalancedSoftmaxCE** |               --                |    **18.65**    |   **15.79**    |      53.19       |    **48.97**    |
|   Input mixup   |          --           |     Class-balanced sampling     |      25.03      |     19.97      |      57.33       |      52.61      |
|   Input mixup   |          --           | Progressively-balanced sampling |      24.70      |     18.68      |      55.50       |      50.67      |
| Manifold mixup  |         CS_CE         |               --                |      26.42      |     21.04      |      68.22       |      58.53      |
| Manifold mixup  |   BalancedSoftmaxCE   |               --                |      23.15      |     18.65      |      56.92       |      52.83      |
| Manifold mixup  |          --           |     Class-balanced sampling     |      27.55      |     22.30      |      63.44       |      58.15      |
| Manifold mixup  |          --           | Progressively-balanced sampling |      26.24      |     21.50      |      60.04       |      56.57      |
|      Remix      |         CS_CE         |               --                |      23.95      |     18.53      |      62.65       |      54.70      |
|      Remix      |   BalancedSoftmaxCE   |               --                |      18.94      |     15.81      |    **51.97**     |      49.10      |
|      Remix      |          --           |     Class-balanced sampling     |      23.87      |     19.80      |      57.66       |      52.71      |
|      Remix      |          --           | Progressively-balanced sampling |      23.81      |     18.49      |      55.66       |      51.25      |

- Findings:
  - **Mixup shows excellent performances when combined with re-balancing methods.**
  - Among the above combinations, input mixup with BalancedSoftmaxCE obtains the best results. **Specially, the two methods,  input mixup  and BalancedSoftmaxCE,  have achieved the best results reported in our paper, which indicates that  BalancedSoftmaxCE is a simple and effective re-weighting method.**
  - Remix obtains comparable results with input mixup, but remix has more hyper-parameters.



### 2. Results of bag of tricks 

|               |        Frist Stage +  Second Stage         |         Third Stage          | CIFAR-10-LT-100 | CIFAR-10-LT-50 | CIFAR-100-LT-100 | CIFAR-100-LT-50 |
| ------------- | :----------------------------------------: | :--------------------------: | :-------------: | :------------: | :--------------: | :-------------: |
| Baseline      |                     --                     |              CE              |      30.12      |     24.81      |      61.76       |      57.65      |
|               | CE+IM+DRS with CAM-based  balance-sampling |              --              |      21.33      |     16.78      |      54.09       |      49.73      |
| Bag of tricks | CE+IM+DRS with CAM-based  balance-sampling | CE+Fine-tuning without mixup |    **20.24**    |   **16.44**    |    **52.17**     |    **47.98**    |

- ``IM`` means ``input mixup``.
- Corresponding CONFIGs at [this link]().

- Bag of tricks contains three stages, and you can reproduce the above results by the following steps:
  - First stage+Second stage:
    - First stage: bash data_parallel_train.sh FIRST-STAGE-CONFIG GPU
    - CAM generation (generating the augmented images):  bash data_parallel_train.sh CAM-GENERATION-CONFIG GPU
    - Second stage: bash data_parallel_train.sh SECOND-STAGE-CONFIG GPU
  - Third stage: bash data_parallel_train.sh THIRD-STAGE-CONFIG GPU

