import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust
import math
from tensorboardX import SummaryWriter
from dataset.inat import INaturalist
from dataset.imagenet import ImageNetLT
from models import resnet_big, resnext
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc
# from torch.models.tensorboard import SummaryWriter
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', default='/DATACENTER/raid5/zjg/imagenet', metavar='DIR')
parser.add_argument('--arch', default='resnext50', choices=['resnet50', 'resnext50'])
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--temp', default=0.1, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[360, 380], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'randstack-randstack'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=1024, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=True, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=True, type=bool,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')


def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'temp', str(args.temp),
         'lr', str(args.lr), args.cl_views, 'noscale'])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 1000 if args.dataset == 'imagenet' else 8142
    if args.arch == 'resnet50':
        model = resnext.BCLModel(name='resnet50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = resnext.BCLModel(name='resnext50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    txt_train = f'/share/home/zjg/BCL/dataset/ImageNet_LT/ImageNet_LT_train.txt' if args.dataset == 'imagenet' \
        else f'/share/home/zjg/BCL/dataset/iNaturalist18/iNaturalist18_train.txt'
    txt_val = f'/share/home/zjg/BCL/dataset/ImageNet_LT/ImageNet_LT_val.txt' if args.dataset == 'imagenet' \
        else f'/share/home/zjg/BCL/dataset/iNaturalist18/iNaturalist18_val.txt'

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randncls),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'randstack-randstack':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False,
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False)

    train_dataset = INaturalist(
        root=args.data,
        txt=txt_train,
        transform=transform_train
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_train,
        transform=transform_train)

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_ce = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)

    # log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    # log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    # with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
    #     f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0

    if args.reload:
        txt_test = f'/share/home/zjg/BCL/dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
            else f'/share/home/zjg/BCL/dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, many, med, few = validate(train_loader, test_loader, model, criterion_ce, 1, args, tf_writer)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1,
                                                                                                   many,
                                                                                                   med,
                                                                                                   few))
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        # acc1, many, medium, few = validate(train_dataset, val_loader, model, criterion_ce, epoch, args,
        #                                    tf_writer)
        acc1, many, med, few = validate(train_loader, val_loader, model, criterion_ce, epoch, args,
                                        tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
        print('Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(best_acc1,
                                                                                                        best_many,
                                                                                                        best_med,
                                                                                                        best_few))
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = torch.cat([inputs[0], inputs[1], inputs[2]], dim=0)
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = targets.shape[0]
        feat_mlp, logits, centers = model(inputs)
        centers = centers[:args.cls_num]
        _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
        features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        scl_loss = criterion_scl(centers, features, targets)
        ce_loss = criterion_ce(logits, targets)
        loss = args.alpha * ce_loss + args.beta * scl_loss

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        acc1 = accuracy(logits, targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, ))  # TODO
            print(output)

        tf_writer.add_scalar('CE loss/train', ce_loss_all.avg, epoch)
        tf_writer.add_scalar('SCL loss/train', scl_loss_all.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)


def validate(train_loader, val_loader, model, criterion_ce, epoch, args, tf_writer=None, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            feat_mlp, logits, centers = model(inputs)
            ce_loss = criterion_ce(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, top1=top1, ))  # TODO
            print(output)

        tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
        tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, total_labels, train_loader,
                                                                acc_per_cls=False)
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'bcl_ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
