import random
#from numpy import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
import torch
from ..registry import PIPELINES
from ...core.visulization import vis_img_tensor
import cv2
import albumentations as A
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
#from mmcv.transforms.utils import cache_randomness

np.random.seed(0)
random.seed(0)

def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class Imgaug:
    """Imgaug augmentation.

    Adds custom transformations from imgaug library.
    Please visit `https://imgaug.readthedocs.io/en/latest/index.html`
    to get more information. Two demo configs could be found in tsn and i3d
    config folder.

    It's better to use uint8 images as inputs since imgaug works best with
    numpy dtype uint8 and isn't well tested with other dtypes. It should be
    noted that not all of the augmenters have the same input and output dtype,
    which may cause unexpected results.

    Required keys are "imgs", "img_shape"(if "gt_bboxes" is not None) and
    "modality", added or modified keys are "imgs", "img_shape", "gt_bboxes"
    and "proposals".

    It is worth mentioning that `Imgaug` will NOT create custom keys like
    "interpolation", "crop_bbox", "flip_direction", etc. So when using
    `Imgaug` along with other mmaction2 pipelines, we should pay more attention
    to required keys.

    Two steps to use `Imgaug` pipeline:
    1. Create initialization parameter `transforms`. There are three ways
        to create `transforms`.
        1) string: only support `default` for now.
            e.g. `transforms='default'`
        2) list[dict]: create a list of augmenters by a list of dicts, each
            dict corresponds to one augmenter. Every dict MUST contain a key
            named `type`. `type` should be a string(iaa.Augmenter's name) or
            an iaa.Augmenter subclass.
            e.g. `transforms=[dict(type='Rotate', rotate=(-20, 20))]`
            e.g. `transforms=[dict(type=iaa.Rotate, rotate=(-20, 20))]`
        3) iaa.Augmenter: create an imgaug.Augmenter object.
            e.g. `transforms=iaa.Rotate(rotate=(-20, 20))`
    2. Add `Imgaug` in dataset pipeline. It is recommended to insert imgaug
        pipeline before `Normalize`. A demo pipeline is listed as follows.
        ```
        pipeline = [
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
            ),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Imgaug', transforms='default'),
            # dict(type='Imgaug', transforms=[
            #     dict(type='Rotate', rotate=(-20, 20))
            # ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        ```

    Args:
        transforms (str | list[dict] | :obj:`iaa.Augmenter`): Three different
            ways to create imgaug augmenter.
    """

    def __init__(self, transforms):
        import imgaug.augmenters as iaa

        if transforms == 'default':
            self.transforms = self.default_transforms()
        elif isinstance(transforms, list):
            assert all(isinstance(trans, dict) for trans in transforms)
            self.transforms = transforms
        elif isinstance(transforms, iaa.Augmenter):
            self.aug = self.transforms = transforms
        else:
            raise ValueError('transforms must be `default` or a list of dicts'
                             ' or iaa.Augmenter object')

        if not isinstance(transforms, iaa.Augmenter):
            self.aug = iaa.Sequential(
                [self.imgaug_builder(t) for t in self.transforms])

    def default_transforms(self):
        """Default transforms for imgaug."""

        return [
            dict(type='Rotate', rotate=(-30, 30)),
            dict(
                type='SomeOf',
                n=(0, 3),
                children=[
                    dict(
                        type='OneOf',
                        children=[
                            dict(type='GaussianBlur', sigma=(0, 0.5)),
                            dict(type='AverageBlur', k=(2, 7)),
                            dict(type='MedianBlur', k=(3, 11))
                        ]),
                    dict(
                        type='OneOf',
                        children=[
                            dict(
                                type='Dropout', p=(0.01, 0.1),
                                per_channel=0.5),
                            dict(
                                type='CoarseDropout',
                                p=(0.03, 0.15),
                                size_percent=(0.02, 0.05),
                                per_channel=0.2),
                        ]),
                    dict(
                        type='AdditiveGaussianNoise',
                        loc=0,
                        scale=(0.0, 0.05 * 255),
                        per_channel=0.5),
                ]),
        ]

    def imgaug_builder(self, cfg):
        """Import a module from imgaug.

        It follows the logic of :func:`build_from_cfg`. Use a dict object to
        create an iaa.Augmenter object.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj:`iaa.Augmenter`: The constructed imgaug augmenter.
        """
        import imgaug.augmenters as iaa

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            obj_cls = getattr(iaa, obj_type)
        elif issubclass(obj_type, iaa.Augmenter):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'children' in args:
            args['children'] = [
                self.imgaug_builder(child) for child in args['children']
            ]

        return obj_cls(**args)

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.aug})'
        return repr_str

    def __call__(self, results):
        assert results['modality'] == 'RGB', 'Imgaug only support RGB images.'
        in_type = results['imgs'][0].dtype.type

        cur_aug = self.aug.to_deterministic()

        results['imgs'] = [
            cur_aug.augment_image(frame) for frame in results['imgs']
        ]
        img_h, img_w, _ = results['imgs'][0].shape

        out_type = results['imgs'][0].dtype.type
        assert in_type == out_type, \
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            from imgaug.augmentables import bbs
            bbox_list = [
                bbs.BoundingBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                for bbox in results['gt_bboxes']
            ]
            bboxes = bbs.BoundingBoxesOnImage(
                bbox_list, shape=results['img_shape'])
            bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
            results['gt_bboxes'] = [[
                max(bbox.x1, 0),
                max(bbox.y1, 0),
                min(bbox.x2, img_w),
                min(bbox.y2, img_h)
            ] for bbox in bbox_aug.items]
            if 'proposals' in results:
                bbox_list = [
                    bbs.BoundingBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                    for bbox in results['proposals']
                ]
                bboxes = bbs.BoundingBoxesOnImage(
                    bbox_list, shape=results['img_shape'])
                bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
                results['proposals'] = [[
                    max(bbox.x1, 0),
                    max(bbox.y1, 0),
                    min(bbox.x2, img_w),
                    min(bbox.y2, img_h)
                ] for bbox in bbox_aug.items]

        results['img_shape'] = (img_h, img_w)

        return results


@PIPELINES.register_module()
class Fuse:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomScale:
    """Resize images by a random scale.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "scale", "resize_size". Required keys in "lazy" is None, added or
    modified key is "interpolation".

    Args:
        scales (tuple[int]): Tuple of scales to be chosen for resize.
        mode (str): Selection mode for choosing the scale. Options are "range"
            and "value". If set to "range", The short edge will be randomly
            chosen from the range of minimum and maximum on the shorter one
            in all tuples. Otherwise, the longer edge will be randomly chosen
            from the range of minimum and maximum on the longer one in all
            tuples. Default: 'range'.
    """

    def __init__(self, scales, mode='range', **kwargs):
        self.mode = mode
        if self.mode not in ['range', 'value']:
            raise ValueError(f"mode should be 'range' or 'value', "
                             f'but got {self.mode}')
        self.scales = scales
        self.kwargs = kwargs

    def select_scale(self, scales):
        num_scales = len(scales)
        if num_scales == 1:
            # specify a fixed scale
            scale = scales[0]
        elif num_scales == 2:
            if self.mode == 'range':
                scale_long = [max(s) for s in scales]
                scale_short = [min(s) for s in scales]
                long_edge = np.random.randint(
                    min(scale_long),
                    max(scale_long) + 1)
                short_edge = np.random.randint(
                    min(scale_short),
                    max(scale_short) + 1)
                scale = (long_edge, short_edge)
            elif self.mode == 'value':
                scale = random.choice(scales)
        else:
            if self.mode != 'value':
                raise ValueError("Only 'value' mode supports more than "
                                 '2 image scales')
            scale = random.choice(scales)

        return scale

    def __call__(self, results):
        scale = self.select_scale(self.scales)
        results['scale'] = scale
        resize = Resize(scale, **self.kwargs)
        results = resize(results)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scales={self.scales}, mode={self.mode})')
        return repr_str


# Note, entity box transfroms are not added to: ThreeCrop, TenCrop,
# MultiGroupCrop.
@PIPELINES.register_module()
class EntityBoxRescale:
    """Rescale the entity box and proposals according to the image shape.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" and
    will be added or modified.

    Args:
        scale_factor (np.ndarray): The scale factor used entity_box rescaling.
    """

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, results):
        scale_factor = np.concatenate([self.scale_factor, self.scale_factor])

        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        results['gt_bboxes'] = gt_bboxes * scale_factor

        if proposals is not None:
            assert proposals.shape[1] == 4, (
                'proposals shape should be in '
                f'(n, 4), but got {proposals.shape}')
            results['proposals'] = proposals * scale_factor

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class EntityBoxCrop:
    """Crop the entity boxes and proposals according to the cropped images.

    Required keys are "proposals", "gt_bboxes", added or modified keys are
    "gt_bboxes". If original "proposals" is not None, "proposals" will be
    modified.

    Args:
        crop_bbox(np.ndarray | None): The bbox used to crop the original image.
    """

    def __init__(self, crop_bbox):
        self.crop_bbox = crop_bbox

    def __call__(self, results):
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']

        if self.crop_bbox is None:
            return results

        x1, y1, x2, y2 = self.crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::2] = np.clip(gt_bboxes[..., 0::2] - x1, 0,
                                        img_w - 1)
        gt_bboxes_[..., 1::2] = np.clip(gt_bboxes[..., 1::2] - y1, 0,
                                        img_h - 1)
        results['gt_bboxes'] = gt_bboxes_

        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::2] = np.clip(proposals[..., 0::2] - x1, 0,
                                            img_w - 1)
            proposals_[..., 1::2] = np.clip(proposals[..., 1::2] - y1, 0,
                                            img_h - 1)
            results['proposals'] = proposals_
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(crop_bbox={self.crop_bbox})'


@PIPELINES.register_module()
class EntityBoxFlip:
    """Flip the entity boxes and proposals with a probability.

    Reverse the order of elements in the given bounding boxes and proposals
    with a specific direction. The shape of them are preserved, but the
    elements are reordered. Only the horizontal flip is supported (seems
    vertical flipping makes no sense). Required keys are "proposals",
    "gt_bboxes", added or modified keys are "gt_bboxes". If "proposals"
    is not None, it will also be modified.

    Args:
        img_shape (tuple[int]): The img shape.
    """

    def __init__(self, img_shape):
        self.img_shape = img_shape
        assert mmcv.is_tuple_of(img_shape, int)

    def __call__(self, results):
        proposals = results['proposals']
        gt_bboxes = results['gt_bboxes']
        img_h, img_w = self.img_shape

        assert gt_bboxes.shape[-1] == 4
        gt_bboxes_ = gt_bboxes.copy()
        gt_bboxes_[..., 0::4] = img_w - gt_bboxes[..., 2::4] - 1
        gt_bboxes_[..., 2::4] = img_w - gt_bboxes[..., 0::4] - 1
        if proposals is not None:
            assert proposals.shape[-1] == 4
            proposals_ = proposals.copy()
            proposals_[..., 0::4] = img_w - proposals[..., 2::4] - 1
            proposals_[..., 2::4] = img_w - proposals[..., 0::4] - 1
        else:
            proposals_ = None

        results['proposals'] = proposals_
        results['gt_bboxes'] = gt_bboxes_
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(img_shape={self.img_shape})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError(f'Size must be an tuple, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        w_size, h_size = self.size
        img_h, img_w = results['img_shape']
        assert h_size <= img_h and w_size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > h_size:
            y_offset = int(np.random.randint(0, img_h - h_size))
        if img_w > w_size:
            x_offset = int(np.random.randint(0, img_w - w_size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = w_size / img_w, h_size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = h_size, w_size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class StereoRandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "imgs" and "img_shape", added or
    modified keys are "imgs", "lazy"; Required keys in "lazy" are "flip",
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not (isinstance(size, tuple) or isinstance(size, list)):
            raise TypeError(f'Size must be an tuple, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if "intrinsics" in results:
            intri = results["intrinsics"][0]
            cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
        _init_lazy_if_proper(results, self.lazy)
        w_size, h_size = self.size
        img_h, img_w = results['img_shape']
        assert h_size <= img_h and w_size <= img_w, "h {} <= {}, w {} <= {}".format(h_size,img_h,w_size,img_w)

        y_offset = 0
        x_offset = 0
        if img_h > h_size:
            y_offset = int(np.random.randint(0, img_h - h_size))
        if img_w > w_size:
            x_offset = int(np.random.randint(0, img_w - w_size))
        loop_num = 0
        while cx-x_offset<=0 or  cy-y_offset<=0:
            if img_h > h_size:
                y_offset = int(np.random.randint(0, img_h - h_size))
            if img_w > w_size:
                x_offset = int(np.random.randint(0, img_w - w_size))
            loop_num +=1
            if loop_num >=100:
                y_offset = (img_h-h_size)//2
                x_offset = (img_w-w_size)//2
                break 
        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = w_size / img_w, h_size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = h_size, w_size

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])

        results['img_shape'] = (new_h, new_w)
        if "intrinsics" in results:
            for i in range(2):

                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-x_offset, cy-y_offset

                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy


        if not self.lazy:
            results['left_imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['left_imgs']
            ]
            results['right_imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['right_imgs']
            ]
            if 'left_depths' in results:
                results['left_depths'] = [
                    img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                    for img in results['left_depths']
                ]
                results['right_depths'] = [
                    img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                    for img in results['right_depths']
                ]
        else:
            raise ValueError
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop:
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "imgs", "img_shape", "crop_bbox" and "lazy",
    added or modified keys are "imgs", "crop_bbox" and "lazy"; Required keys
    in "lazy" are "flip", "crop_bbox", added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                keys,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.keys = keys
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            for key in self.keys:
                results[key] = [
                    img[top:bottom, left:right] for img in results[key]
                ]
                if "intrinsics" in results:
                    assert len(results["intrinsics"].shape) == 3, "intrinsics should be 3D matrix"
                    len_intrinsics = results["intrinsics"].shape[0]
                    assert len_intrinsics == 1 or len_intrinsics == 2
                    for i in range(len_intrinsics):
                        intri = results["intrinsics"][i]
                        cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                        # for random cropping, focal will not change, princeple center will minirs left-up point
                        cx, cy = cx-left, cy-top
                        results["intrinsics"][i][0][2] = cx
                        results["intrinsics"][i][1][2] = cy
                

        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop:
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image weight and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "img_shape", "lazy" and "scales". Required keys in "lazy" are
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): Weight and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center". If set to 13, the cropping bbox will
            append another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        base_size = min(img_h, img_w)
        crop_sizes = [int(base_size * s) for s in self.scales]

        candidate_sizes = []
        for i, h in enumerate(crop_sizes):
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    candidate_sizes.append([w, h])

        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        crop_w, crop_h = crop_size

        if self.random_crop:
            x_offset = random.randint(0, img_w - crop_w)
            y_offset = random.randint(0, img_h - crop_h)
        else:
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            candidate_offsets = [
                (0, 0),  # upper left
                (4 * w_step, 0),  # upper right
                (0, 4 * h_step),  # lower left
                (4 * w_step, 4 * h_step),  # lower right
                (2 * w_step, 2 * h_step),  # center
            ]
            if self.num_fixed_crops == 13:
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                candidate_offsets.extend(extra_candidate_offsets)
            x_offset, y_offset = random.choice(candidate_offsets)

        new_h, new_w = crop_h, crop_w

        results['crop_bbox'] = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['img_shape'] = (new_h, new_w)
        results['scales'] = self.scales

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            results['imgs'] = [
                img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str

@PIPELINES.register_module()
class DispDepthTransform:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, disp2depth=True, keys=[], fake_focal=False):
        self.fake_focal = fake_focal
        self.disp2depth = disp2depth
        if keys==[]:
            keys=["left","right"]
        self.keys = keys 

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if "focal" not in results:
            self.fake_focal = True
            results["focal"] = 380
            results["baseline"] = 1

        if self.disp2depth:
            # disp map to depth map
            if "left_disps" in results and "left" in self.keys:
                results["left_depths"] = []
                for i in range(len(results["left_disps"])):
                    results["left_disps"][i][results["left_disps"][i]==0] = 1e-6
                    results["left_depths"].append(results["focal"]*results["baseline"]/(results["left_disps"][i])) 
                    #print("max depth >>", max(results["left_depths"][i][results["left_disps"][i]>1].reshape(-1)))
                    # max depth in sceneflow (disp>1) mostly < 600
            if "right_disps" in results and "right" in self.keys:
                results["right_depths"] = []
                for i in range(len(results["right_disps"])):
                    results["right_disps"][i][results["right_disps"][i]==0] = 1e-6
                    results["right_depths"].append(results["focal"]*results["baseline"]/(results["right_disps"][i])) 
        else:
            # depth map to disp map
            if "left_depths" in results and "left" in self.keys:
                results["left_disps"] = []
                for i in range(len(results["left_depths"])):
                    results["left_depths"][i][results["left_depths"][i]==0] = 1e-6
                    results["left_disps"].append(results["focal"]*results["baseline"]/(results["left_depths"][i])) 
            if "right_depths" in results and "right" in self.keys:
                results["right_disps"] = []
                for i in range(len(results["right_depths"])):
                    results["right_depths"][i][results["right_depths"][i]==0] = 1e-6
                    results["right_disps"].append(results["focal"]*results["baseline"]/(results["right_depths"][i])) 
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'keys={self.keys}, disp2depth={self.disp2depth}, '
                    f'fake_focal={self.fake_focal} ' )
        return repr_str


@PIPELINES.register_module()
class StereoResize:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if "intrinsics" in results:
            intrinsics = results["intrinsics"]
            for i in range(2):
                intri = intrinsics[i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy, fx, fy = cx*self.scale_factor[0], cy*self.scale_factor[1], \
                                fx*self.scale_factor[0], fy*self.scale_factor[1]
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy
                results["intrinsics"][i][0][0] = fx
                results["intrinsics"][i][1][1] = fy
        if "focal" in results:
            results["focal"] *= self.scale_factor[0]

        results['left_imgs'] = [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in results['left_imgs']
        ]
        if "right_imgs" in results.keys() and results["right_imgs"] is not None:
            results['right_imgs'] = [
                mmcv.imresize(
                    img, (new_w, new_h), interpolation=self.interpolation)
                for img in results['right_imgs']
            ]
        if "left_pred_depths" in results.keys() and results["left_pred_depths"] is not None:
            for depth in results['left_pred_depths']:
                assert len(depth.shape) == 2, "depth has to be loaded with HxW shape"
            #print("before >> ", results['left_depths'][0])
            results['left_pred_depths'] = [ F.interpolate(torch.FloatTensor(depth.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
                            for depth in results['left_pred_depths'] ]
        if "right_pred_depths" in results.keys()  and results["right_pred_depths"] is not None :
            for depth in results['right_pred_depths']:
                assert len(depth.shape) == 2, "depth has to be loaded with HxW shape"
            results['right_pred_depths'] = [ F.interpolate(torch.FloatTensor(depth.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
                            for depth in results['right_pred_depths'] ]
        if "left_depths" in results.keys() and results["left_depths"] is not None:
            for depth in results['left_depths']:
                assert len(depth.shape) == 2, "depth has to be loaded with HxW shape"
            #print("before >> ", results['left_depths'][0])
            results['left_depths'] = [ F.interpolate(torch.FloatTensor(depth.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
                            for depth in results['left_depths'] ]
            #print("after >> ", results['left_depths'][0])
            #results['left_depths'] = [
            #100*mmcv.imresize(
            #    img.astype(np.float64)/100, (new_w, new_h), interpolation=self.interpolation)
            #for img in results['left_depths']
            #]
        if "right_depths" in results.keys()  and results["right_depths"] is not None :
            for depth in results['right_depths']:
                assert len(depth.shape) == 2, "depth has to be loaded with HxW shape"
            results['right_depths'] = [ F.interpolate(torch.FloatTensor(depth.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
                            for depth in results['right_depths'] ]

        if "left_disps" in results.keys() and results["left_disps"] is not None:
            for disp in results["left_disps"]:
                assert len(disp.shape) == 2, "the disparity has to be loaded with HxW shape"
            # resizing disparity maps needs to * x dim scale_factor
            results['left_disps'] = [self.scale_factor[0] * F.interpolate(torch.FloatTensor(disp.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
            for disp in results['left_disps']
            ]
        if "right_disps" in results.keys()  and results["right_disps"] is not None :
            for disp in results["right_disps"]:
                assert len(disp.shape) == 2, "the disparity has to be loaded with HxW shape"
            results['right_disps'] = [self.scale_factor[0] * F.interpolate(torch.FloatTensor(disp.copy()).unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest", ).squeeze().numpy()
            for disp in results['right_disps']
            ]
        if "left_masks" in results.keys() and results["left_masks"] is not None:
            results['left_masks'] = [
            mmcv.imresize(
                img.astype(np.float64), (new_w, new_h), interpolation='nearest')
            for img in results['left_masks']
            ]
      
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RotateMonoImg(object):
    
    def __init__(self, ):
        pass
    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
         
        if "intrinsics" in results:
            intrinsics = results["intrinsics"]
            for i in range(2):
                intri = intrinsics[i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # rotate 90 degree
                results["intrinsics"][i][0][2] = cy
                results["intrinsics"][i][1][2] = cx
                results["intrinsics"][i][0][0] = fy
                results["intrinsics"][i][1][1] = fx
                results["intrinsics"][i] = results["intrinsics"][i].astype(np.float64)
        if "focal" in results:
            results["focal"]  = fy.astype(np.float64)

        results['left_imgs'] = [
            np.rot90(img, -1, (0,1)).astype(np.float64) for img in results['left_imgs']
        ]
        return results

@PIPELINES.register_module()
class Resize:
    """Resize images to a specific size.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "lazy",
    "resize_size". Required keys in "lazy" is None, added or modified key is
    "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 keys,
                 scale,
                 keep_ratio=True,
                 ratio_range=None,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.keys = keys
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy
        self.ratio_range = ratio_range
        if self.ratio_range is not None:
            assert isinstance(self.ratio_range, tuple)
            assert len(self.ratio_range) == 2
            assert self.ratio_range[0] <= self.ratio_range[1]
            min_ratio, max_ratio = self.ratio_range
            ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            self.scale = (self.scale[0] * ratio, self.scale[1] * ratio)

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            for k in self.keys:
                results[k] = [
                    mmcv.imresize(
                        img, (new_w, new_h), interpolation=self.interpolation)
                    for img in results[k]
                ]
                if 'disp' in k:
                    results[k] = [
                        img * self.scale_factor[0] for img in results[k]
                    ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_rescale = EntityBoxRescale(self.scale_factor)
            results = entity_box_rescale(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomRescale:
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmcv.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.
    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "lazy" and "flip_direction". Required keys in "lazy" is
    None, added or modified key are "flip" and "flip_direction". The Flip
    augmentation should be placed after any cropping / reshaping augmentations,
    to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.lazy = lazy

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                for i, img in enumerate(results['imgs']):
                    mmcv.imflip_(img, self.direction)
                lt = len(results['imgs'])
                for i in range(0, lt, 2):# flow with even indexes are x_flow, which need to be
                    # inverted when doing horizontal flip
                    if modality == 'Flow':
                        results['imgs'][i] = mmcv.iminvert(results['imgs'][i])

            else:
                results['imgs'] = list(results['imgs'])
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            entity_box_flip = EntityBoxFlip(results['img_shape'])
            results = entity_box_flip(results)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_rgb=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            results['imgs'] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class StereoNormalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, keys=['left_imgs', 'right_imgs'], to_rgb=False, to_gray=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.to_gray = to_gray
        self.adjust_magnitude = adjust_magnitude
        self.keys = keys

    def __call__(self, results):
        #if results['img_norm_cfg']['to_rgb'] == True: # already to rgb when loading image
        #    self.to_rgb = False
        for key in self.keys:
            n = len(results[key])
            if self.to_gray:
                for i, img in enumerate(results[key]):
                    if results['img_norm_cfg']['to_rgb'] == True:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, axis=-1)
                    results[key][i] = img
            if len(results[key][0].shape) == 2:
                for i, img in enumerate(results[key]):
                    img = np.expand_dims(img, axis=-1)
                    results[key][i] = img
            h, w, c = results[key][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results[key]):
                imgs[i] = img
            results["raw_"+key] = imgs.copy()
            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, to_rgb=self.to_rgb)
            results[key] = imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results
         

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_rgb={self.to_rgb}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """Randomly distort the brightness, contrast, saturation and hue of images,
    and add PCA based noise into images.

    Note: The input images should be in RGB channel order.

    Code Reference:
    https://gluon-cv.mxnet.io/_modules/gluoncv/data/transforms/experimental/image.html
    https://mxnet.apache.org/api/python/docs/_modules/mxnet/image/image.html#LightingAug

    If specified to apply color space augmentation, it will distort the image
    color space by changing brightness, contrast and saturation. Then, it will
    add some random distort to the images in different color channels.
    Note that the input images should be in original range [0, 255] and in RGB
    channel sequence.

    Required keys are "imgs", added or modified keys are "imgs", "eig_val",
    "eig_vec", "alpha_std" and "color_space_aug".

    Args:
        color_space_aug (bool): Whether to apply color space augmentations. If
            specified, it will change the brightness, contrast, saturation and
            hue of images, then add PCA based noise to images. Otherwise, it
            will directly add PCA based noise to images. Default: False.
        alpha_std (float): Std in the normal Gaussian distribution of alpha.
        eig_val (np.ndarray | None): Eigenvalues of [1 x 3] size for RGB
            channel jitter. If set to None, it will use the default
            eigenvalues. Default: None.
        eig_vec (np.ndarray | None): Eigenvectors of [3 x 3] size for RGB
            channel jitter. If set to None, it will use the default
            eigenvectors. Default: None.
    """

    def __init__(self,
                 color_space_aug=False,
                 alpha_std=0.1,
                 eig_val=None,
                 eig_vec=None):
        if eig_val is None:
            # note that the data range should be [0, 255]
            self.eig_val = np.array([55.46, 4.794, 1.148], dtype=np.float32)
        else:
            self.eig_val = eig_val

        if eig_vec is None:
            self.eig_vec = np.array([[-0.5675, 0.7192, 0.4009],
                                     [-0.5808, -0.0045, -0.8140],
                                     [-0.5836, -0.6948, 0.4203]],
                                    dtype=np.float32)
        else:
            self.eig_vec = eig_vec

        self.alpha_std = alpha_std
        self.color_space_aug = color_space_aug

    @staticmethod
    def brightness(img, delta):
        """Brightness distortion.

        Args:
            img (np.ndarray): An input image.
            delta (float): Delta value to distort brightness.
                It ranges from [-32, 32).

        Returns:
            np.ndarray: A brightness distorted image.
        """
        if np.random.rand() > 0.5:
            img = img + np.float32(delta)
        return img

    @staticmethod
    def contrast(img, alpha):
        """Contrast distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to distort contrast.
                It ranges from [0.6, 1.4).

        Returns:
            np.ndarray: A contrast distorted image.
        """
        if np.random.rand() > 0.5:
            img = img * np.float32(alpha)
        return img

    @staticmethod
    def saturation(img, alpha):
        """Saturation distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to distort the saturation.
                It ranges from [0.6, 1.4).

        Returns:
            np.ndarray: A saturation distorted image.
        """
        if np.random.rand() > 0.5:
            gray = img * np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = np.sum(gray, 2, keepdims=True)
            gray *= (1.0 - alpha)
            img = img * alpha
            img = img + gray
        return img

    @staticmethod
    def hue(img, alpha):
        """Hue distortion.

        Args:
            img (np.ndarray): An input image.
            alpha (float): Alpha value to control the degree of rotation
                for hue. It ranges from [-18, 18).

        Returns:
            np.ndarray: A hue distorted image.
        """
        if np.random.rand() > 0.5:
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]],
                          dtype=np.float32)
            tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]],
                            dtype=np.float32)
            ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]],
                             dtype=np.float32)
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            t = np.array(t, dtype=np.float32)
            img = np.dot(img, t)
        return img

    def __call__(self, results):
        imgs = results['imgs']
        out = []
        if self.color_space_aug:
            bright_delta = np.random.uniform(-32, 32)
            contrast_alpha = np.random.uniform(0.6, 1.4)
            saturation_alpha = np.random.uniform(0.6, 1.4)
            hue_alpha = np.random.uniform(-18, 18)
            jitter_coin = np.random.rand()
            for img in imgs:
                img = self.brightness(img, delta=bright_delta)
                if jitter_coin > 0.5:
                    img = self.contrast(img, alpha=contrast_alpha)
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                else:
                    img = self.saturation(img, alpha=saturation_alpha)
                    img = self.hue(img, alpha=hue_alpha)
                    img = self.contrast(img, alpha=contrast_alpha)
                out.append(img)
        else:
            out = imgs

        # Add PCA based noise
        alpha = np.random.normal(0, self.alpha_std, size=(3, ))
        rgb = np.array(
            np.dot(self.eig_vec * alpha, self.eig_val), dtype=np.float32)
        rgb = rgb[None, None, ...]

        results['imgs'] = [img + rgb for img in out]
        results['eig_val'] = self.eig_val
        results['eig_vec'] = self.eig_vec
        results['alpha_std'] = self.alpha_std
        results['color_space_aug'] = self.color_space_aug

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'color_space_aug={self.color_space_aug}, '
                    f'alpha_std={self.alpha_std}, '
                    f'eig_val={self.eig_val}, '
                    f'eig_vec={self.eig_vec})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop:
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            results['imgs'] = [
                img[top:bottom, left:right] for img in results['imgs']
            ]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_crop = EntityBoxCrop(results['crop_bbox'])
            results = entity_box_crop(results)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str

@PIPELINES.register_module()
class StereoCenterCrop:
    """Crop the center area from images.

    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox", "lazy" and "img_shape". Required keys in "lazy" is
    "crop_bbox", added or modified key is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = crop_size
        self.lazy = lazy
        #if not mmcv.is_tuple_of(self.crop_size, int):
        #    raise TypeError(f'Crop_size must be int or tuple of int, '
        #                    f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
 
        img_h, img_w = results['img_shape']
        if len(self.crop_size) ==4:
            top = int(self.crop_size[0] * img_h)
            bottom = int(self.crop_size[1] * img_h)
            left = int(self.crop_size[2] * img_w)
            right = int(self.crop_size[3] * img_w)

        else:
            crop_w, crop_h = self.crop_size
            left = (img_w - crop_w) // 2
            top = (img_h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
        new_h, new_w = bottom - top, right - left
        results['crop_bbox'] = np.array([left, top, right, bottom])
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_x_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)
        if "intrinsics" in results:
            for i in range(2):
                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-left, cy-top
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy

        if not self.lazy:
            results['left_imgs'] = [
                img[top:bottom, left:right]
                for img in results['left_imgs']
            ]
            if 'right_imgs' in results:
                results['right_imgs'] = [
                    img[top:bottom, left:right]
                    for img in results['right_imgs']
                ]
            if 'left_depths' in results:
                results['left_depths'] = [
                    img[top:bottom, left:right]
                    for img in results['left_depths']
                ]
            if 'right_depths' in results:
                results['right_depths'] = [
                    img[top:bottom, left:right]
                    for img in results['right_depths']
                ]

         
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop:
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop:
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class MultiGroupCrop:
    """Randomly crop the images into several groups.

    Crop the random region with the same given crop_size and bounding box
    into several groups.
    Required keys are "imgs", added or modified keys are "imgs", "crop_bbox"
    and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
        groups(int): Number of groups.
    """

    def __init__(self, crop_size, groups):
        self.crop_size = _pair(crop_size)
        self.groups = groups
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError('Crop size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

        if not isinstance(groups, int):
            raise TypeError(f'Groups must be int, but got {type(groups)}.')

        if groups <= 0:
            raise ValueError('Groups must be positive.')

    def __call__(self, results):
        """Performs the MultiGroupCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        img_h, img_w = imgs[0].shape[:2]
        crop_w, crop_h = self.crop_size

        img_crops = []
        crop_bboxes = []
        for _ in range(self.groups):
            x_offset = np.random.randint(0, img_w - crop_w)
            y_offset = np.random.randint(0, img_h - crop_h)

            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            img_crops.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(crop_size={self.crop_size}, '
                    f'groups={self.groups})')
        return repr_str


@PIPELINES.register_module()
class AudioAmplify:
    """Amplify the waveform.

    Required keys are "audios", added or modified keys are "audios",
    "amplify_ratio".

    Args:
        ratio (float): The ratio used to amplify the audio waveform.
    """

    def __init__(self, ratio):
        if isinstance(ratio, float):
            self.ratio = ratio
        else:
            raise TypeError('Amplification ratio should be float.')

    def __call__(self, results):
        """Perfrom the audio amplification.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        assert 'audios' in results
        results['audios'] *= self.ratio
        results['amplify_ratio'] = self.ratio

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(ratio={self.ratio})'
        return repr_str


@PIPELINES.register_module()
class MelSpectrogram:
    """MelSpectrogram. Transfer an audio wave into a melspectogram figure.

    Required keys are "audios", "sample_rate", "num_clips", added or modified
    keys are "audios".

    Args:
        window_size (int): The window size in milisecond. Default: 32.
        step_size (int): The step size in milisecond. Default: 16.
        n_mels (int): Number of mels. Default: 80.
        fixed_length (int): The sample length of melspectrogram maybe not
            exactly as wished due to different fps, fix the length for batch
            collation by truncating or padding. Default: 128.
    """

    def __init__(self,
                 window_size=32,
                 step_size=16,
                 n_mels=80,
                 fixed_length=128):
        if all(
                isinstance(x, int)
                for x in [window_size, step_size, n_mels, fixed_length]):
            self.window_size = window_size
            self.step_size = step_size
            self.n_mels = n_mels
            self.fixed_length = fixed_length
        else:
            raise TypeError('All arguments should be int.')

    def __call__(self, results):
        """Perform MelSpectrogram transformation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError('Install librosa first.')
        signals = results['audios']
        sample_rate = results['sample_rate']
        n_fft = int(round(sample_rate * self.window_size / 1000))
        hop_length = int(round(sample_rate * self.step_size / 1000))
        melspectrograms = list()
        for clip_idx in range(results['num_clips']):
            clip_signal = signals[clip_idx]
            mel = librosa.feature.melspectrogram(
                y=clip_signal,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.n_mels)
            if mel.shape[0] >= self.fixed_length:
                mel = mel[:self.fixed_length, :]
            else:
                mel = np.pad(
                    mel, ((0, mel.shape[-1] - self.fixed_length), (0, 0)),
                    mode='edge')
            melspectrograms.append(mel)

        results['audios'] = np.array(melspectrograms)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(window_size={self.window_size}), '
                    f'step_size={self.step_size}, '
                    f'n_mels={self.n_mels}, '
                    f'fixed_length={self.fixed_length})')
        return repr_str


### augmentation for stereo matching, 
### refer to 
### codes partly from MMsegmentation


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, left_img, right_img):
        """Brightness distortion."""
        if np.random.randint(2):
            beta = random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
            return (self.convert(  left_img, beta=beta), self.convert(  right_img,  beta=beta))
        return (left_img, right_img)

    def contrast(self, left_img, right_img):
        """Contrast distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            return self.convert(left_img,  alpha=alpha), self.convert(right_img,  alpha=alpha)
        return left_img, right_img

    def saturation(self, left_img, right_img):
        """Saturation distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.saturation_lower,
                                     self.saturation_upper)
            left_img = mmcv.rgb2bgr(left_img)
            left_img = mmcv.bgr2hsv(left_img)
            left_img[:, :, 1] = self.convert(
                left_img[:, :, 1],
                alpha=alpha)
            left_img = mmcv.hsv2bgr(left_img)
            left_img = mmcv.bgr2rgb(left_img)
            ##----
            right_img = mmcv.rgb2bgr(right_img)
            right_img = mmcv.bgr2hsv(right_img)
            right_img[:, :, 1] = self.convert(
                right_img[:, :, 1],
                alpha=alpha)
            right_img = mmcv.hsv2bgr(right_img)
            right_img = mmcv.bgr2rgb(right_img)
        return left_img, right_img

    def hue(self, left_img, right_img):
        """Hue distortion."""
        if np.random.randint(2):
            rand_bias = np.random.randint(-self.hue_delta, self.hue_delta)
            left_img = mmcv.rgb2bgr(left_img)
            left_img = mmcv.bgr2hsv(left_img)
            left_img[:, :,
                0] = (left_img[:, :, 0].astype(int) +rand_bias) % 180
            left_img = mmcv.hsv2bgr(left_img)
            left_img = mmcv.bgr2rgb(left_img)
            #---
            right_img = mmcv.rgb2bgr(right_img)
            right_img = mmcv.bgr2hsv(right_img)
            right_img[:, :,
                0] = (right_img[:, :, 0].astype(int) +rand_bias) % 180
            right_img = mmcv.hsv2bgr(right_img)
            right_img = mmcv.bgr2rgb(right_img)
        return left_img, right_img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        #if "left_imgs" in results.keys():
        left_imgs = results["left_imgs"]
        if "right_imgs" in results.keys():
            right_imgs = results["right_imgs"]
        else:
            right_imgs = results["left_imgs"]
        left_res = []
        right_res = []
        for l_img, r_img in zip(left_imgs,right_imgs):
            l_img, r_img = self._subcall(l_img, r_img)
            left_res.append(l_img)
            right_res.append(r_img)
        results["left_imgs"] = left_res
        results["right_imgs"] = right_res
        return results

    def _subcall(self, left_img, right_img):
        #img = results['img']
        # random brightness
        left_img, right_img = self.brightness(left_img, right_img)
        

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(0,2)
        if mode == 1:
            left_img, right_img = self.contrast(left_img, right_img)

        # random saturation
        left_img, right_img = self.saturation(left_img, right_img)
        # random hue
        left_img, right_img = self.hue(left_img, right_img)
        # random contrast
        if mode == 0:
            left_img, right_img = self.contrast(left_img, right_img)

        #results['img'] = img
        return left_img, right_img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

@PIPELINES.register_module()
class PhotoMetricDistortionDifferent(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, left_img):
        """Brightness distortion."""
        if np.random.randint(2):
            beta = random.uniform(-self.brightness_delta,
                                            self.brightness_delta)
            return self.convert(  left_img, beta=beta)
        return left_img

    def contrast(self, left_img):
        """Contrast distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            return self.convert(left_img,  alpha=alpha)
        return left_img

    def saturation(self, left_img):
        """Saturation distortion."""
        if np.random.randint(2):
            alpha = random.uniform(self.saturation_lower,
                                     self.saturation_upper)
            left_img = mmcv.rgb2bgr(left_img)
            left_img = mmcv.bgr2hsv(left_img)
            left_img[:, :, 1] = self.convert(
                left_img[:, :, 1],
                alpha=alpha)
            left_img = mmcv.hsv2bgr(left_img)
            left_img = mmcv.bgr2rgb(left_img)
          
        return left_img

    def hue(self, left_img):
        """Hue distortion."""
        if np.random.randint(2):
            rand_bias = np.random.randint(-self.hue_delta, self.hue_delta)
            left_img = mmcv.rgb2bgr(left_img)
            left_img = mmcv.bgr2hsv(left_img)
            left_img[:, :,
                0] = (left_img[:, :, 0].astype(int) +rand_bias) % 180
            left_img = mmcv.hsv2bgr(left_img)
            left_img = mmcv.bgr2rgb(left_img)

        return left_img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        #if "left_imgs" in results.keys():
        left_imgs = results["left_imgs"]
        if "right_imgs" in results.keys():
            right_imgs = results["right_imgs"]
        else:
            right_imgs = results["left_imgs"]
        left_res = []
        right_res = []
        for l_img, r_img in zip(left_imgs,right_imgs):
            l_img = self._subcall(l_img)
            r_img = self._subcall(r_img)
            left_res.append(l_img)
            right_res.append(r_img)
        results["left_imgs"] = left_res
        results["right_imgs"] = right_res
        return results

    def _subcall(self, img):
        #img = results['img']
        # random brightness
        img = self.brightness(img)
        

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(0,2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)
        # random hue
        img = self.hue(img)
        # random contrast
        if mode == 0:
            img = self.contrast(img)

        #results['img'] = img
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str



@PIPELINES.register_module()
class StereoResize2(object):
    """Resize images & disparity & depth.
    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.
    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:
    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)
    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)
    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)
    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing. W, H
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
        min_size (int, optional): The minimum size for input and the shape
            of the image and seg map will not be less than ``min_size``.
            As the shape of model input is fixed like 'SETR' and 'BEiT'.
            Following the setting in these models, resized images must be
            bigger than the crop size in ``slide_inference``. Default: None
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 min_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.
        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.
        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.
        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.
        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, img, scale):
        """Resize images with ``results['scale']``."""
        img, w_scale, h_scale = mmcv.imresize(
            img, scale, return_scale=True)  
        return img


    def _resize_depth_disp(self, disp, scale, w_scale_factor=1.0):
        """ disp 2D, scale: (w, h); use 'nearest' mode to resize disp map, 
        avoid the blur of object edges.
        only disparity map needs to multiply w_scale_factor"""
        
        return w_scale_factor * F.interpolate(torch.FloatTensor(disp.copy()).squeeze().unsqueeze(0).unsqueeze(0),
                        size=(scale[1],scale[0]), mode="nearest",).squeeze().numpy()


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/resize_before.png", results["left_imgs"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/resize_before_disp.png", results["left_disps"][0],)
        if 'scale' not in results.keys():
            self._random_scale(results)
        
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        if "img_shape" not in results.keys():
            img_h, img_w = results["left_imgs"][0].shape[:2]
        else:
            img_h, img_w = results['img_shape']

        if self.keep_ratio:
            if self.min_size is not None:
                # if given minsize, update the resize scale, when scale < min_size
                # TODO: Now 'min_size' is an 'int' which means the minimum
                # shape of images is (min_size, min_size, 3). 'min_size'
                # with tuple type will be supported, i.e. the width and
                # height are not equal.
                if min(results['scale']) < self.min_size:
                    new_short = self.min_size
                else:
                    new_short = min(results['scale'])

                h, w = img_h, img_w
                if h > w:
                    new_h, new_w = new_short  / w * h, new_short
                else:
                    new_h, new_w = new_short, new_short / h * w 
                results['scale'] = (new_w, new_h)

            new_w, new_h = mmcv.rescale_size((img_w, img_h), results["scale"])
        else:
            new_w, new_h = results["scale"] # don't keep ratio w/h

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)
        results['img_shape'] = (new_h, new_w)
        results["pad_shape"] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if "intrinsics" in results.keys():
            intrinsics = results["intrinsics"]
            for i in range(2):
                intri = intrinsics[i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy, fx, fy = cx*self.scale_factor[0], cy*self.scale_factor[1], \
                                fx*self.scale_factor[0], fy*self.scale_factor[1]
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy
                results["intrinsics"][i][0][0] = fx
                results["intrinsics"][i][1][1] = fy
        if "focal" in results.keys():
            results["focal"] *= self.scale_factor[0]


        if "left_imgs" in results.keys():
            results["left_imgs"] =  [self._resize_img(img, results["scale"]) for img in results["left_imgs"]] 
        if "right_imgs" in results.keys():
            results["right_imgs"] =  [self._resize_img(img, results["scale"]) for img in results["right_imgs"]] 
        #self._resize_img(results)
        #self._resize_seg(results)
        if "left_depths" in results.keys():
            results["left_depths"] = [self._resize_depth_disp(depth, results["scale"]) for depth in results["left_depths"] ]
        if "right_depths" in results.keys():
            results["right_depths"] = [self._resize_depth_disp(depth, results["scale"]) for depth in results["right_depths"] ]
        if "left_disps" in results.keys():
            results["left_disps"] = [self._resize_depth_disp(depth, results["scale"], results["scale_factor"][0]) for depth in results["left_disps"] ]
        if "right_disps" in results.keys():
            results["right_disps"] = [self._resize_depth_disp(depth, results["scale"], results["scale_factor"][0]) for depth in results["right_disps"] ]
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/resize_after.png", results["left_imgs"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/resize_after_disp.png", results["left_disps"][0],)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str

@PIPELINES.register_module()
class SceneFlowTestCrop(object):
    """  crop  to  right botoom corner
    Args:
        
    """

    def __init__(self, crop_size=[512,960], keys=[]):
        #assert crop_size[0] > 0 and crop_size[1] > 0
        self.keys = keys
        self.crop_size = crop_size
 

    # kb cropping
    def cropping(self, img):
        h_im, w_im = img.shape[:2]
        self.margin_top = int(h_im - self.crop_size[0])
        self.margin_left = 0 #int((w_im - self.crop_size[1]))

        img = img[self.margin_top: self.margin_top + self.crop_size[0],
                  self.margin_left: self.margin_left + self.crop_size[1]]
        return img

    def __call__(self, results):
        assert len(self.keys) > 0, " set left_imgs, right_imgs, left_depths, right_depths"
        if "intrinsics" in results:
            intri = results["intrinsics"][0]
            cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]

        for k in self.keys:
            for t in range(len(results[k])):
                results[k][t] = self.cropping(results[k][t])
        
        if "intrinsics" in results:
            for i in range(2):
                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-self.margin_left, cy-self.margin_top
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class StereoTopLeftCrop(object):
    def __init__(self, crop_size, keys=[]):
        self.crop_size = crop_size # h, w 
        self.keys = keys
        assert len(keys) > 0
    def __call__(self, results):
        for k in self.keys:
            seq_imgs = results[k]
            h, w = seq_imgs[0].shape[:2]
            min_h = min(h, self.crop_size[0])
            min_w = min(w, self.crop_size[1])
            new_seq_imgs = []
            for img in seq_imgs:
                new_seq_imgs.append(img[:min_h, :min_w, ...])
        
            results[k] = new_seq_imgs
        
        return results

@PIPELINES.register_module()
class KITTIKBCrop(object):
    """ kb cropping for kitti supervised training 
    Args:
        
    """

    def __init__(self, keys=[], crop_size=[352,1216]):
        #assert crop_size[0] > 0 and crop_size[1] > 0
        self.keys = keys
        self.crop_size=crop_size
        if isinstance(crop_size, list):
            assert crop_size[1]>=crop_size[0]
 

    # kb cropping
    def cropping(self, img):
        h_im, w_im = img.shape[:2]
        
        if isinstance(self.crop_size, (list,tuple)):
            self.margin_top = int(h_im - self.crop_size[0])
            self.margin_left = int((w_im - self.crop_size[1]) / 2)
            #self.margin_left = int((w_im - self.crop_size[1]) / 2)
        
            img = img[self.margin_top: self.margin_top + self.crop_size[0],
                    self.margin_left: self.margin_left + self.crop_size[1]]
        elif isinstance(self.crop_size, int):
            self.margin_top = int(h_im - self.crop_size)
            img = img[self.margin_top: self.margin_top + self.crop_size,]
        else:
            raise NotImplementedError
        return img

    def __call__(self, results):
        assert len(self.keys) > 0, " set left_imgs, right_imgs, left_depths, right_depths"

        for k in self.keys:
            for t in range(len(results[k])):
                results[k][t] = self.cropping(results[k][t])
        
        if "intrinsics" in results:
            assert len(results["intrinsics"].shape) == 3, "intrinsics should be 3D matrix"
            len_intrinsics = results["intrinsics"].shape[0]
            assert len_intrinsics == 1 or len_intrinsics == 2
            for i in range(len_intrinsics):
                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-self.margin_left, cy-self.margin_top
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class KITTICenterCrop(object):
    """ center cropping for kitti supervised training 
    Args:
        
    """

    def __init__(self, crop_size, keys=[]):
        #assert crop_size[0] > 0 and crop_size[1] > 0
        # crop_size [top, down, left, right]
        self.keys = keys
        self.crop_size = crop_size
 

    # kb cropping
    def cropping(self, img):
        h_im, w_im = img.shape[:2]
        crop = np.array([self.crop_size[0] * h_im //16*16,  self.crop_size[1] * h_im//16*16,   
                    self.crop_size[2] * w_im//16*16,   self.crop_size[3] * w_im//16*16]).astype(np.int32)
        self.margin_top, self.margin_left = crop[0], crop[2]
        img = img[crop[0]:crop[1],
                  crop[2]:crop[3],...]
        return img

    def __call__(self, results):
        assert len(self.keys) > 0, " set left_imgs, right_imgs, left_depths, right_depths"
        if "intrinsics" in results:
            intri = results["intrinsics"][0]
            cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]

        for k in self.keys:
            for t in range(len(results[k])):
                results[k][t] = self.cropping(results[k][t])
        
        if "intrinsics" in results:
            for i in range(2):
                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-self.margin_left, cy-self.margin_top
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class StereoRandomCrop2(object):
    """Random crop the image & seg.
    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        zeros_disp_max_ratio (float): The maximum ratio that <1 disp values could
            occupy.
    """

    def __init__(self, crop_size, zeros_disp_max_ratio=1., ignore_val=1, random_shift=False, 
                    random_shift_range=(0,10)):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.zeros_disp_max_ratio = zeros_disp_max_ratio
        self.ignore_val = ignore_val
        self.random_shift = random_shift
        self.random_shift_range = random_shift_range

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # firstly get the crops bbox
        img = results['left_imgs'][0]
        #cv2.imwrite("/home/ziliu/vis/dataaugmentation/crop_before.png", img)
        #cv2.imwrite("/home/ziliu/vis/dataaugmentation/crop_before_disp.png", results["left_disps"][0])
        crop_bbox = self.get_crop_bbox(img)

        if "intrinsics" in results:
            assert len(results["intrinsics"].shape) == 3, "intrinsics should be 3D matrix"
            len_intrinsics = results["intrinsics"].shape[0]
            assert len_intrinsics == 1 or len_intrinsics == 2
            for i in range(len_intrinsics):
                intri = results["intrinsics"][i]
                cx,cy,fx,fy = intri[0][2], intri[1][2], intri[0][0], intri[1][1]
                # for random cropping, focal will not change, princeple center will minirs left-up point
                cx, cy = cx-crop_bbox[2], cy-crop_bbox[0]
                results["intrinsics"][i][0][2] = cx
                results["intrinsics"][i][1][2] = cy

        """ 
        if "left_depths" in results.keys():
            if self.zeros_disp_max_ratio < 1.:
                # Repeat 10 times
                for _ in range(10):
                    disp_temp = self.crop(results['left_depths'][0], crop_bbox)
                    disp_count = np.ones_like(disp_temp)
                    zeros_disp_count = disp_count[disp_temp > self.ignore_val]
                    if np.sum(zeros_disp_count) / np.sum(disp_count) < self.zeros_disp_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)
        if "left_disps" in results.keys():
            if self.zeros_disp_max_ratio < 1.:
                # Repeat 10 times
                for _ in range(10):
                    disp_temp = self.crop(results['left_disps'][0], crop_bbox)
                    disp_count = np.ones_like(disp_temp)
                    zeros_disp_count = disp_count[disp_temp < self.ignore_val]
                    if np.sum(zeros_disp_count) / np.sum(disp_count) < self.zeros_disp_max_ratio:
                        break
                    crop_bbox = self.get_crop_bbox(img)
        """
        """ 
        right_view_shift = 0
        crop_bbox_R = list(crop_bbox).copy()
        if self.random_shift and random.randint(1,100)<30:
            right_view_shift = random.randint(self.random_shift_range[0], self.random_shift_range[1])
            crop_bbox_R[-1] += right_view_shift # x2
            crop_bbox_R[-2] += right_view_shift # x1
            max_w = results["left_imgs"][0].shape[1]
            if crop_bbox_R[-1] > max_w: # h,w,3
                offset = crop_bbox_R[-1] - max_w
                crop_bbox_R[-1] = crop_bbox_R[-1] - offset
                assert crop_bbox_R[-1] == max_w
                crop_bbox_R[-2] = crop_bbox_R[-2] - offset
        crop_bbox_R = tuple(crop_bbox_R)
        """
        # crop the image
        #if "left_imgs" in results.keys():
        results["left_imgs"] = [self.crop(img, crop_bbox) for img in results["left_imgs"]]
        img_shape = results["left_imgs"][0].shape
        results['img_shape'] = img_shape

        if "right_imgs" in results.keys():
            results["right_imgs"] = [self.crop(img, crop_bbox) for img in results["right_imgs"]]
        # crop disp & depth 
        if "left_depths" in results.keys():
            assert self.random_shift == False
            results["left_depths"] = [self.crop(img, crop_bbox) for img in results["left_depths"]]
        if "right_depths" in results.keys():
            assert self.random_shift == False
            results["right_depths"] = [self.crop(img, crop_bbox) for img in results["right_depths"]]
        if "left_disps" in results.keys():
            results["left_disps"] = [self.crop(img, crop_bbox)  for img in results["left_disps"]]
        if "right_disps" in results.keys():
            results["right_disps"] = [self.crop(img, crop_bbox) for img in results["right_disps"]]
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/crop_after.png", results["left_imgs"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/crop_after_disp.png", results["left_disps"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/crop_after_depth.png", results["left_depths"][0],)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

@PIPELINES.register_module()
class StereoPad(object):
    """Pad the image & mask.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        disp_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 disp_pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.disp_pad_val = disp_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val)
        return padded_img

    def _pad_img_torch(self, img):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = F.pad(torch.FloatTensor(img.copy()).permute(2,0,1), \
                ((self.size[1]-img.shape[1])//2, (self.size[1]-img.shape[1])-(self.size[1]-img.shape[1])//2,  \
                 (self.size[0]-img.shape[0])//2, (self.size[0]-img.shape[0])-(self.size[0]-img.shape[0])//2), mode='replicate')
        return padded_img.permute(1,2,0).numpy()
    
    def _pad_disp_depth(self, disp):
        """Pad depth/disp according to ``results['pad_shape']``."""
        # pad shape is obtained in self._pad_img()
        return mmcv.impad(
                disp,
                shape=self.size,
                pad_val=self.disp_pad_val)
    def _pad_disp_depth_torch(self, img):
        """Pad depth/disp according to ``results['pad_shape']``."""
        padded_img = F.pad(torch.FloatTensor(img.copy()).unsqueeze(0), \
               ((self.size[1]-img.shape[1])//2, (self.size[1]-img.shape[1])-(self.size[1]-img.shape[1])//2,  \
                 (self.size[0]-img.shape[0])//2, (self.size[0]-img.shape[0])-(self.size[0]-img.shape[0])//2), mode='replicate')
        return padded_img.squeeze().numpy()
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/pad_before.png", results["left_imgs"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/pad_before_disp.png", results["left_disps"][0],)
        results["left_imgs"] = [self._pad_img_torch(img) for img in results["left_imgs"]]
        #results['pad_shape'] = results["left_imgs"][0].shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        if "right_imgs" in results.keys():
            results["right_imgs"] = [self._pad_img_torch(img) for img in results["right_imgs"]]
        if "left_depths" in results.keys():
            results["left_depths"] = [self._pad_disp_depth_torch(disp )for disp in results["left_depths"]]
        if "right_depths" in results.keys():
            results["right_depths"] = [self._pad_disp_depth_torch(disp )for disp in results["right_depths"]]
        if "left_disps" in results.keys():
            results["left_disps"] = [self._pad_disp_depth_torch(disp,  )for disp in results["left_disps"]]
        if "right_disps" in results.keys():
            results["right_disps"] = [self._pad_disp_depth_torch(disp,  )for disp in results["right_disps"]]
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/pad_after.png", results["left_imgs"][0],)
        #cv2.imwrite( "/home/ziliu/vis/dataaugmentation/pad_after_disp.png", results["left_disps"][0],)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class VerticalCutDepth(object):
    """"
    keys: the left or right view to use cutdepth
    step: 
    """
    def __init__(self, keys=[], step=4):
        self.keys = keys
        self.step = step

    def __call__(self, results ):
        if len(self.keys)==0:
            if "left_depths" in results:
                self.keys.append("left")
            if "right_depths" in results:
                self.keys.append("right")
        H,W,C = results["left_imgs"][0].shape
        if results["count_cutdepth"] % self.step == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))
            for k in self.keys:
                for i in range(len(results[f"{k}_imgs"])):
                    results[f"{k}_imgs"][i][:, l:l+w, 0] = results[f"{k}_depths"][i][:, l:l+w]
                    results[f"{k}_imgs"][i][:, l:l+w, 1] = results[f"{k}_depths"][i][:, l:l+w]
                    results[f"{k}_imgs"][i][:, l:l+w, 2] = results[f"{k}_depths"][i][:, l:l+w]
        return results

@PIPELINES.register_module()
class VerticalCutDisp(object):
    """"
    keys: the left or right view to use cutdepth
    step: 
    """
    def __init__(self, keys=[], step=4):
        self.keys = keys
        self.step = step

    def __call__(self, results ):
        if len(self.keys)==0:
            if "left_disps" in results:
                self.keys.append("left")
            if "right_disps" in results:
                self.keys.append("right")
        
        H,W,C = results["left_imgs"][0].shape
        if results["count_cutdepth"] % self.step == 0:
            alpha = random.random()
            beta = random.random()
            p = 0.75

            l = int(alpha * W)
            w = int(max((W - alpha * W) * beta * p, 1))
            for k in self.keys:
                for i in range(len(results[f"{k}_imgs"])):
                    results[f"{k}_imgs"][i][:, l:l+w, 0] = results[f"{k}_disps"][i][:, l:l+w]
                    results[f"{k}_imgs"][i][:, l:l+w, 1] = results[f"{k}_disps"][i][:, l:l+w]
                    results[f"{k}_imgs"][i][:, l:l+w, 2] = results[f"{k}_disps"][i][:, l:l+w]
        return results

@PIPELINES.register_module()
class HorizontalFlip(object):
    def __init__(self, mono=False, p=0.5):
        self.mono = mono
        self.p = p

    def __call__(self, results):
        if np.random.randint(2):
            return results
        if not self.mono:
            assert ("right_depths" in results and "left_depths" in results) or ("left_disps" in results and "right_disps" in results) 
        
        use_disp = False
        if "left_disps" in results:
            use_disp = True
        use_depth = False
        if "left_depths" in results:
            use_depth = True
        if self.mono:
            for i in range(len(results["left_imgs"])):
                results["left_imgs"][i] = self.apply(results["left_imgs"][i])
                if use_disp:
                    results["left_disps"][i] = self.apply(results["left_disps"][i])
                if use_depth:
                    results["left_depths"][i] = self.apply(results["left_depths"][i])
        else: # stereo pairs are inveresed 
            new_left_imgs = []
            new_left_depths = []
            new_left_disps = []
            new_right_imgs = []
            new_right_depths= []
            new_right_disps = []
            #vis_img_tensor(torch.FloatTensor(results["left_imgs"][0]).permute(2,0,1), "/home/ziliu/vis/aug","leftbefore",False)
            #vis_img_tensor(torch.FloatTensor(results["right_imgs"][0]).permute(2,0,1), "/home/ziliu/vis/aug","rightbefore",False)
            for i in range(len(results["left_imgs"])):
                new_right_imgs.append(self.apply(results["left_imgs"][i]))
                if use_disp:
                    new_right_disps.append(self.apply(results["left_disps"][i]))
                if use_depth:
                    new_right_depths.append(self.apply(results["left_depths"][i]))
            
            for i in range(len(results["right_imgs"])):
                new_left_imgs.append(self.apply(results["right_imgs"][i]))
                if use_disp:
                    new_left_disps.append(self.apply(results["right_disps"][i]))
                if use_depth:
                    new_left_depths.append(self.apply(results["right_depths"][i]))
            
            results["right_imgs"] = new_right_imgs
            if use_depth:
                results["right_depths"] = new_right_depths
            if use_disp:
                results["right_disps"] = new_right_disps
            results["left_imgs"] = new_left_imgs
            if use_depth:
                results["left_depths"] = new_left_depths
            if use_disp:
                results["left_disps"] = new_left_disps
            #vis_img_tensor(torch.FloatTensor(results["left_imgs"][0]).permute(2,0,1), "/home/ziliu/vis/aug","leftafter", False)
            #vis_img_tensor(torch.FloatTensor(results["right_imgs"][0]).permute(2,0,1), "/home/ziliu/vis/aug","rightafter", False)
        return results

    def hflip(self, img: np.ndarray):
        return np.ascontiguousarray(img[:, ::-1, ...])

    def hflip_cv2(self, img: np.ndarray):
        return cv2.flip(img, 1)

    def apply(self, img: np.ndarray, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return self.hflip_cv2(img)

        return self.hflip(img)


@PIPELINES.register_module()
class RandomBrightnessContrast(object):
    MAX_VALUES_BY_DTYPE = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 65535,
        np.dtype("uint32"): 4294967295,
        np.dtype("float32"): 1.0,
    }


    def __init__(self, brightness_limit=0.2,
        contrast_limit=0.2,
        brightness_by_max=True,
        always_apply=False,
        p=0.5,):
        self.brightness_limit = (-brightness_limit, brightness_limit)
        self.contrast_limit = (-contrast_limit, contrast_limit)
        self.brightness_by_max = brightness_by_max
        self.always_apply = always_apply
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results
        alpha= 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        
        if "left_imgs" in results:
            for i in range(len(results["left_imgs"])):
                if results["left_imgs"][0].dtype == np.uint8:
                    results["left_imgs"][i] = self._brightness_contrast_adjust_uint(results["left_imgs"][i], alpha, beta, self.brightness_by_max)
                else:
                    results["left_imgs"][i] = self._brightness_contrast_adjust_non_uint(results["left_imgs"][i], alpha, beta, self.brightness_by_max)

        if "right_imgs" in results:
            for i in range(len(results["right_imgs"])):
                if results["left_imgs"][0].dtype == np.uint8:
                    results["right_imgs"][i] = self._brightness_contrast_adjust_uint(results["right_imgs"][i], alpha, beta, self.brightness_by_max)
                else:
                    results["right_imgs"][i] = self._brightness_contrast_adjust_non_uint(results["right_imgs"][i], alpha, beta, self.brightness_by_max)
        
        return results

    def _brightness_contrast_adjust_uint(self, img, alpha=1, beta=0, beta_by_max=False):
        dtype = np.dtype("uint8")

        max_value = self.MAX_VALUES_BY_DTYPE[dtype]

        lut = np.arange(0, max_value + 1).astype("float32")

        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += (alpha * beta) * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img


    def _brightness_contrast_adjust_non_uint(self, img, alpha=1, beta=0, beta_by_max=False):
        dtype = img.dtype
        img = img.astype("float32")

        if alpha != 1:
            img *= alpha
        if beta != 0:
            if beta_by_max:
                max_value = self.MAX_VALUES_BY_DTYPE[dtype]
                img += beta * max_value
            else:
                img += beta * np.mean(img)
        return img

@PIPELINES.register_module()
class RandomGamma(object):

    def __init__(self, gamma_limit=(80, 120), eps=None, p=0.5):
        self.gamma_limit = gamma_limit
        self.eps = eps 
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results
        gamma = random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        if "left_imgs" in results:
            for i in range(len(results["left_imgs"])):
                results["left_imgs"][i] = self.gamma_transform(results["left_imgs"][i], gamma)
        if "right_imgs" in results:
            for i in range(len(results["right_imgs"])):
                results["right_imgs"][i] = self.gamma_transform(results["right_imgs"][i], gamma)
        return results
        
    def gamma_transform(self, img, gamma):
        if img.dtype == np.uint8:
            table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
            img = cv2.LUT(img, table.astype(np.uint8))
        else:
            img = np.power(img, gamma)

        return img

def is_rgb_image(image: np.ndarray):
    return len(image.shape) == 3 and image.shape[-1] == 3

def is_grayscale_image(image: np.ndarray):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


@PIPELINES.register_module()
class HueSaturationValue(object):

    def __init__(self, hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.5,):
        self.hue_shift_limit = (-hue_shift_limit, hue_shift_limit)
        self.sat_shift_limit = (-sat_shift_limit, sat_shift_limit)
        self.val_shift_limit = (-val_shift_limit, val_shift_limit)
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results
        hue_shift = random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
        sat_shift = random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
        val_shift = random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
        
        if "left_imgs" in results:
            for i in range(len(results["left_imgs"])):
                results["left_imgs"][i] = self.shift_hsv(results["left_imgs"][i], hue_shift, sat_shift, val_shift)
        if "right_imgs" in results:
            for i in range(len(results["right_imgs"])):
                results["right_imgs"][i] = self.shift_hsv(results["right_imgs"][i], hue_shift, sat_shift, val_shift)
        return results

    def shift_hsv(self, img, hue_shift, sat_shift, val_shift):
        if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
            return img

        is_gray = is_grayscale_image(img)
        if is_gray:
            if hue_shift != 0 or sat_shift != 0:
                hue_shift = 0
                sat_shift = 0
                Warning(
                    "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                    "Set them to 0 or use RGB image"
                )
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.dtype == np.uint8:
            img = self._shift_hsv_uint8(img, hue_shift, sat_shift, val_shift)
        else:
            img = self._shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift)

        if is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        return img

    def _shift_hsv_uint8(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            lut_hue = np.arange(0, 256, dtype=np.int16)
            lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
            hue = cv2.LUT(hue, lut_hue)

        if sat_shift != 0:
            lut_sat = np.arange(0, 256, dtype=np.int16)
            lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
            sat = cv2.LUT(sat, lut_sat)

        if val_shift != 0:
            lut_val = np.arange(0, 256, dtype=np.int16)
            lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
            val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img


    def _shift_hsv_non_uint8(self, img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            hue = cv2.add(hue, hue_shift)
            hue = np.mod(hue, 360)  # OpenCV fails with negative values

        if sat_shift != 0:
            sat = self.clip(cv2.add(sat, sat_shift), dtype, 1.0)

        if val_shift != 0:
            val = self.clip(cv2.add(val, val_shift), dtype, 1.0)

        img = cv2.merge((hue, sat, val))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img


    def clip(img: np.ndarray, dtype: np.dtype, maxval: float):
        return np.clip(img, 0, maxval).astype(dtype)



from PIL import Image, ImageEnhance


@PIPELINES.register_module()
class Augmentor(object):
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, results):
        left_img, right_img, left_disp = results["left_imgs"][0], \
        results["right_imgs"][0], results["left_disps"][0]
        if "right_disps" in results:
            right_disp = results["right_disps"][0]
            if self.rng.binomial(1, 0.5):
                left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
                left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
            left_disp[left_disp == np.inf] = 0
            right_disp[right_disp == np.inf] = 0

        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )
        if "right_disps" in results:
            right_disp = (
                cv2.resize(
                    right_disp,
                    None,
                    fx=resize_scale,
                    fy=resize_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
                * resize_scale
            )

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        if "right_disps" in results:
            right_disp = cv2.warpAffine(
                right_disp,
                M,
                (self.image_width, self.image_height),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]
        results["left_imgs"][0], results["right_imgs"][0], results["left_disps"][0] \
            = left_img, right_img, left_disp
        if "right_disps" in results:
            results["right_disps"][0]  = right_disp
        return results


from mmcv.image import adjust_brightness, adjust_color, adjust_contrast

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    Args:
        img (ndarray): Image to be adjust with BGR order.
        value (float): the amount of shift in H channel and must be in the
            interval [-0.5, 0.5].. 0.5 and -0.5 give complete reversal of hue
            channel in HSV space in positive and negative direction
            respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will
            give an image with complementary colors while 0 gives the original
            image.
    Returns:
        ndarray: The hue-adjusted image.
    """
    if hue_factor is None:
        return img
    else:
        assert hue_factor >= -0.5 and hue_factor <= 0.5
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(np.int_) + int(hue_factor * 180.) +
                        180) % 180
        img = mmcv.hsv2bgr(img)
        return img


def adjust_gamma(img, gamma=1.0):
    """Using gamma correction to process the image.
    Args:
        img (ndarray): Image to be adjusted. uint8 datatype.
        gamma (float or int): Gamma value used in gamma correction. gamma is a
            positive value. Note: gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter. Default: 1.0.
    """

    assert isinstance(gamma, float) or isinstance(gamma, int)
    assert gamma > 0

    assert img.dtype == 'uint8'

    table = ((np.arange(256) / 255.) ** gamma * (255 + 1 - 1e-3))\
        .astype('uint8')

    adjusted_img = mmcv.lut_transform(np.array(img, dtype=np.uint8), table)

    return adjusted_img

from glob import glob
from skimage import color, io
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F
import warnings

def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0,1), keepdims=True)# + 1
    reference_mean = np.mean(reference_image_lab, axis=(0,1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev/reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l,a,b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb

class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

@PIPELINES.register_module()
class FlowAugmentor(object):
    def __init__(
        self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, 
                 yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

        print("FlowAugmentor: ", self.__dict__)

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow


    def __call__(self, results):
        img1, img2, disp = results["left_imgs"][0], \
        results["right_imgs"][0], results["left_disps"][0]
        #flow = np.expand_dims(flow, 2)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        results["left_imgs"][0], results["right_imgs"][0], results["left_disps"][0] \
            = img1, img2, flow[:,:,0]
        return results



@PIPELINES.register_module()
class StereoColorJitter:
    """Randomly change the brightness, contrast, saturation and hue of
    an image.
    Args:
        asymmetric_prob (float): the probability to do color jitter for two
            images asymmetrically.
        brightness (float, tuple):  How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast (float, tuple):  How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non negative numbers.
        saturation (float, tuple):  How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non negative numbers.
        hue (float, tuple): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given
            [min, max]. Should have 0<= hue <= 0.5 or
            -0.5 <= min <= max <= 0.5.
    """

    def __init__(self,
                 img_keys,
                 asymmetric_prob=0.,
                 brightness=0.,
                 contrast=0.,
                 saturation=0.,
                 hue=0.):
        assert isinstance(
            asymmetric_prob, float
        ), f'asymmetric_prob must be float, but got {type(asymmetric_prob)}'
        self.asymmetric_prob = asymmetric_prob
        self.img_keys = img_keys

        self._brightness = self._check_input(brightness, 'brightness')
        self._contrast = self._check_input(contrast, 'contrast')
        self._saturation = self._check_input(saturation, 'saturation')
        self._hue = self._check_input(
            hue, 'hue', center=0., bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _get_param(self):

        fn_idx = np.random.permutation(4)
        b = None if self._brightness is None else np.random.uniform(
            self._brightness[0], self._brightness[1])
        c = None if self._contrast is None else np.random.uniform(
            self._contrast[0], self._contrast[1])
        s = None if self._saturation is None else np.random.uniform(
            self._saturation[0], self._saturation[1])
        h = None if self._hue is None else np.random.uniform(
            self._hue[0], self._hue[1])

        return fn_idx, b, c, s, h

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, (float, int)):

            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        elif isinstance(value, (tuple, list)) and len(value) == 2:

            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        else:
            raise TypeError(
                f'{name} should be a single number or a list/tuple with '
                f'length 2, but got {value}.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def color_jitter(self, img_list):
        
        fn_idx, brightness, contrast, saturation, hue = self._get_param()
        for k in range(len(img_list)):
            img = img_list[k] # left or right images
            for i in fn_idx:
                if i == 0 and brightness:
                    img = [adjust_brightness(i, brightness) for i in img]
                if i == 1 and contrast:
                    img = [adjust_contrast(i, contrast) for i in img]
                if i == 2 and saturation:
                    img = [adjust_color(i, saturation) for i in img]
                if i == 3 and hue:
                    img = [adjust_hue(i, hue) for i in img]
            img_list[k] = img
        return img_list

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img_keys = self.img_keys
        imgs = []
        for k in img_keys:
            imgs.append(results[k])
        asym = np.random.rand()
        # asymmetric
        if asym < self.asymmetric_prob:
            imgs_ = []
            for i in imgs:
                i = self.color_jitter([i])[0]
                imgs_.append(i)
            imgs = imgs_
        else:
            # symmetric
            imgs = self.color_jitter(imgs)
        for i, k in enumerate(img_keys):
            results[k] = imgs[i]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'asymmetric_prob={self.asymmetric_prob}, '
                     f'brightness_range={self._brightness}, '
                     f'contrast_range={self._contrast}, '
                     f'saturation_range={self._saturation}, '
                     f'hue_range={self._hue}')
        return repr_str


@PIPELINES.register_module()
class PreLoadImageVolume(object):
    def __init__(self, max_disp=192, mask_template=None):
        self.max_disp = max_disp
        self.mask_template = mask_template


    def build_image_volume_roll(self, leftImage, rightImage, max_disp ):
        #device = leftImage.device
        H, W, C = leftImage.shape
        #D = max_disp
        if self.mask_template is None or self.mask_template.shape[1:3]!=leftImage.shape[:2]:
            print("re-init mask template")
            print("leftImage.shape: ", leftImage.shape)
            print("mask_template.shape: ", self.mask_template.shape if self.mask_template is not None else "None")
            self.mask_template = np.ones((max_disp, H, W, 2*C), dtype=np.float32)
            for i in range(max_disp):
                self.mask_template[i, :, 0:i, :] = 0
        image_volume = [np.concatenate((leftImage, np.roll(rightImage, i, axis=1)), axis=2) for i in range(max_disp)]
        image_volume = np.stack(image_volume, axis=0) # D, H, W, 2*C
        
        if self.mask_template is not None:
            image_volume = image_volume * self.mask_template
        return image_volume
    
    def __call__(self, results):
        seq_n = len(results['left_imgs'])
        seq_image_volume = []
        for i in range(seq_n):
            left_images, right_image = results["left_imgs"][i], results['right_imgs'][i]
            seq_image_volume.append(self.build_image_volume_roll(left_images, right_image, self.max_disp))
        results['image_volumes'] = seq_image_volume
        return results
        

@PIPELINES.register_module()
class RGB2Gray(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            gray_key = "gray_" + key
            results[gray_key] = []
            for i in range(len(results[key])):
                results[gray_key].append(cv2.cvtColor(results[key][i], cv2.COLOR_RGB2GRAY))
        return results

@PIPELINES.register_module()
class RandomResize(object):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[1] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        keys,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        keep_ratio=True,
        lazy=False,
        interpolation='bilinear',
        #resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        self.keys = keys
        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.lazy = lazy
        self.interpolation = interpolation

        #self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        ## create a empty Reisize object
        #self.resize = self.resize_cfg.update({'scale': 0,})

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    #@cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type
        of ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        if isinstance(self.scale[0], int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif isinstance(self.scale[0], tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

        return scale

    def _resize(self, results ):
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            for k in self.keys:
                results[k] = [
                    mmcv.imresize(
                        img.copy(), (new_w, new_h), interpolation=self.interpolation)
                    for img in results[k]
                ]
                if "disp" in k:
                    results[k] = [img * self.scale_factor[0] for img in results[k]]
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            entity_box_rescale = EntityBoxRescale(self.scale_factor)
            results = entity_box_rescale(results)
        return results

    def __call__(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        results['scale'] = self._random_scale()
        self.scale = results['scale']
        results = self._resize(results)
        return results
