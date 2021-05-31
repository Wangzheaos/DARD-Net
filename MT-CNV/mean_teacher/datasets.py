import torchvision.transforms as transforms

from . import data
from .utils import export
from PIL import Image


@export
def CNV():
    train_channel_stats = dict(mean=[0.20, 0.20, 0.20],
                               std=[0.08, 0.08, 0.08])
    eval_channel_stats = dict(mean=[0.19, 0.19, 0.19],
                              std=[0.09, 0.09, 0.09])
    cent_channel_stats = dict(mean=[0.24, 0.24, 0.24],
                              std=[0.12, 0.12, 0.12])

    train_transformation = data.TransformTwice(transforms.Compose([
        # transforms.Resize((80, 80), Image.ANTIALIAS),
        data.RandomTranslateWithReflect(4),  #old=12
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(3),  # 随机旋转在（-6， +6）
        transforms.ColorJitter(brightness=(0.9, 1.2), contrast=(1, 1.3)),  # 修改图片的亮度、对比度、饱和度
        transforms.ToTensor(),
        transforms.Normalize(**train_channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        # transforms.Resize((80, 80), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(**eval_channel_stats)
    ])
    cent_transformation = transforms.Compose([
        # transforms.Resize((32, 32), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(**cent_channel_stats)
    ])
    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'cent_transformation': cent_transformation,
        'datadir': 'data-local/images/CNV',
        'num_classes': 6
    }


@export
def ruxian():
    train_channel_stats = dict(mean=[0.20, 0.20, 0.20],
                               std=[0.08, 0.08, 0.08])
    eval_channel_stats = dict(mean=[0.19, 0.19, 0.19],
                              std=[0.09, 0.09, 0.09])
    cent_channel_stats = dict(mean=[0.24, 0.24, 0.24],
                              std=[0.12, 0.12, 0.12])

    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.Resize((80, 80), Image.ANTIALIAS),
        data.RandomTranslateWithReflect(12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(6),  # 随机旋转在（-6， +6）
        transforms.ColorJitter(brightness=(0.9, 1.2), contrast=(1, 1.3)),  # 修改图片的亮度、对比度、饱和度
        transforms.ToTensor(),
        transforms.Normalize(**train_channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize((80, 80), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(**eval_channel_stats)
    ])
    cent_transformation = transforms.Compose([
        transforms.Resize((80, 80), Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize(**cent_channel_stats)
    ])
    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'cent_transformation': cent_transformation,
        'datadir': 'data-local/images/ruxian',
        'num_classes': 4
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914],
                         std=[0.2470])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/cifar/cifar10/by-image',
        'num_classes': 10
    }


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }
