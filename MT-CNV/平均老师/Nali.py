import math
import os
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
# from nvidia.dali.backend import oss
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from nvidia.dali.types import DALIDataType


class CommonPipeline(Pipeline):
    def __init__(self,
                 batch_size,
                 num_workers,
                 image_size=(256, 256),
                 image_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 image_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 random_aspect_ratio=[0.75, 1.333333],
                 train=True,
                 device_id=0,
                 shard_id=0,
                 seed=0,
                 decoder_device='mixed',
                 **kwargs):
        super(CommonPipeline, self).__init__(batch_size,
                                             num_workers,
                                             device_id,
                                             seed=seed + shard_id,
                                             **kwargs)
        self.train = train
        self.dali_device = 'gpu' if decoder_device == 'mixed' else 'cpu',
        self.decoder_device = decoder_device

        if train:
            self.decode = ops.ImageDecoder(device=decoder_device,
                                           output_type=types.RGB)
            self.coin = ops.CoinFlip(probability=0.5)
            self.resize = ops.Resize(
                device='gpu' if decoder_device == 'mixed' else 'cpu',
                interp_type=types.INTERP_TRIANGULAR)
        else:
            self.decode = ops.ImageDecoder(device=decoder_device,
                                           output_type=types.RGB)
            self.resize = ops.Resize(
                device='gpu' if decoder_device == 'mixed' else 'cpu',
                resize_x=image_size[1],
                resize_y=image_size[0],
                interp_type=types.INTERP_TRIANGULAR)

        assert isinstance(image_size, tuple) or isinstance(image_size, list)
        assert len(image_size) == 2

        self.cnmp = ops.CropMirrorNormalize(
            device='gpu' if decoder_device == 'mixed' else 'cpu',
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=image_mean,
            std=image_std)
        self.augmentations = []

    def base_define_graph(self, inputs, targets):
        inputs = self.decode(inputs)
        for augment in self.augmentations:
            inputs = augment(inputs)
        inputs = self.resize(inputs)
        if self.dali_device == 'gpu':
            inputs = inputs.gpu()
            targets = targets.gpu()
        if self.train:
            inputs = self.cnmp(inputs, mirror=self.coin())
        else:
            inputs = self.cnmp(inputs)
        return inputs, targets

    def define_graph(self):
        raise NotImplementedError

    def add_augmentations(self, ops):
        """Add augmentation list
        Args:
            ops: list of DALI ops that will perform image augmentations on decoded image data.
        """
        self.augmentations + ops


class ExternalSourcePipeline(CommonPipeline):
    def __init__(self, sampler_iterator,
                 batch_size,
                 num_workers,
                 image_size=(256, 256),
                 image_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                 image_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                 random_aspect_ratio=[0.75, 1.333333],
                 train=True,
                 device_id=0,
                 shard_id=0,
                 seed=0,
                 decoder_device='mixed',
                 **kwargs):
        super(ExternalSourcePipeline, self).__init__(
            batch_size, num_workers, image_size, image_mean, image_std,
            random_aspect_ratio, train, device_id, shard_id,
            seed, decoder_device, **kwargs)
        self.sampler_iterator = sampler_iterator
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        return self.base_define_graph(self.jpegs, self.labels)

    def iter_setup(self):
        (images, labels) = next(self.sampler_iterator)
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def reset_sampler_iterator(self, sampler_iterator):
        self.sampler_iterator = sampler_iterator


class ClassificationIterator(DALIGenericIterator):
    def __init__(self, sampler, pipelines, size, fill_last_batch=True, last_batch_padded=False):
        super(ClassificationIterator, self).__init__(pipelines, ["data", "label"], size, auto_reset=False,
                                                     fill_last_batch=fill_last_batch,
                                                     dynamic_shape=False,
                                                     last_batch_padded=last_batch_padded)
        self.sampler = sampler

    def __len__(self):
        return math.ceil(self._size / self.batch_size)

    def reset(self, epoch):
        self.sampler.set_epoch(epoch)
        for p in self._pipes:
            p.reset_sampler_iterator(iter(self.sampler))
        super(ClassificationIterator, self).reset()


def make_dataloader(sampler, pipeline, num_shards, train):
    pipeline.build()
    print('pipeline build successful')
    assert len(sampler) % num_shards == 0
    size = len(sampler) / num_shards
    print('pipeline size{}'.format(size))
    if train:
        return ClassificationIterator(sampler=sampler, pipelines=pipeline,
                                      size=size,
                                      fill_last_batch=True,
                                      last_batch_padded=False)
    else:
        return ClassificationIterator(sampler=sampler, pipelines=pipeline,
                                      size=size,
                                      fill_last_batch=False,
                                      last_batch_padded=True)
