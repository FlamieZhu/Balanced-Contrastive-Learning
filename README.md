# Balanced Contrastive Learning for Long-Tailed Visual Recognition

This is a Pytorch implementation of the [BCL paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf):
```
@inproceedings{zhu2022balanced,
  title={Balanced Contrastive Learning for Long-Tailed Visual Recognition},
  author={Zhu, Jianggang and Wang, Zheng and Chen, Jingjing and Chen, Yi-Ping Phoebe and Jiang, Yu-Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6908--6917},
  year={2022}
}
```
## Abstract
Real-world data typically follow a long-tailed distribution, where a few majority categories occupy most of the data while most minority categories contain a limited number of samples. Classification models minimizing cross-entropy struggle to represent and classify the tail classes. Although the problem of learning unbiased classifiers has been well studied, methods for representing imbalanced data are under-explored. In this paper, we focus on representation learning for imbalanced data. Recently, supervised contrastive learning has shown promising performance on balanced data recently. However, through our theoretical analysis, we find that for long-tailed data, it fails to form a regular simplex which is an ideal geometric configuration for representation learning. To correct the optimization behavior of SCL and further improve the performance of long-tailed visual recognition, we propose a novel loss for balanced contrastive learning (BCL). Compared with SCL, we have two improvements in BCL: class-averaging, which balances the gradient contribution of negative classes; class-complement, which allows all classes to appear in every mini-batch. The proposed balanced contrastive learning (BCL) method satisfies the condition of forming a regular simplex and assists the optimization of cross-entropy. Equipped with BCL, the proposed two-branch framework can obtain a stronger feature representation and achieve competitive performance on long-tailed benchmark datasets such as CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist2018.

## Requirement
- pytorch>=1.6.0
- torchvision
- tensorboardX

## Usage
### ImageNet-LT
```
python main.py --data /share/common/ImageDatasets/imagenet_2012 \
  --lr 0.1 -p 200 --epochs 90 \
  --arch resnext50 --use-norm True \
  --wd 5e-4 --cos True \
  --cl_views sim-sim
```
