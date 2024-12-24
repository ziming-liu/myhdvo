import io
from operator import rshift
import os
import os.path as osp
import shutil
from unittest import result
import warnings
import cv2 

import mmcv
import numpy as np
from PIL import Image
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..registry import PIPELINES

np.random.seed(3407)


@PIPELINES.register_module()
class LoadHVULabel:
    """Convert the HVU label from dictionaries to torch tensors.

    Required keys are "label", "categories", "category_nums", added or modified
    keys are "label", "mask" and "category_mask".
    """

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def __call__(self, results):
        """Convert the label dictionary to 3 tensors: "label", "mask" and
        "category_mask".

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        #print(total_frames)
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)
        #print(clip_offsets)
        #print(frame_inds)
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets
        #print(frame_inds)
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')
        #print(frame_inds)
        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        #print(frame_inds)
        #exit()
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class VOSampleFrames(SampleFrames):
    """Sample frames from the video.

    Required keys are "filename", "total_frames", "start_index" , added or
    modified keys are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        base_offsets = np.arange(self.num_clips) * avg_interval
        clip_offsets = base_offsets 
        

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets.astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        #print(total_frames)
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)
        #print(clip_offsets)
        #print(frame_inds)
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets
        #print(frame_inds)
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')
        #print(frame_inds)
        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        #print(frame_inds)
        #exit()
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str




@PIPELINES.register_module()
class UntrimmedSampleFrames:
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10. That is to say, by default,
            there are at least 10 clips for one input sample in test mode.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 sample_range=64,
                 num_sample_positions=10,
                 temporal_jitter=False,
                 out_of_bound_opt='loop',
                 test_mode=False):
        super().__init__(
            clip_len,
            frame_interval,
            num_clips,
            temporal_jitter,
            out_of_bound_opt=out_of_bound_opt,
            test_mode=test_mode)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)

        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        results['frame_inds'] = np.array(frame_inds, dtype=np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    @staticmethod
    def _get_train_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    @staticmethod
    def _get_val_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (obj): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=np.int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'body_segments={self.body_segments}, '
                    f'aug_segments={self.aug_segments}, '
                    f'aug_ratio={self.aug_ratio}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_interval={self.test_interval}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PyAVInit:
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend=disk)'
        return repr_str


@PIPELINES.register_module()
class PyAVDecode:
    """Using pyav to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max indice to make early stop
        max_inds = max(results['frame_inds'])
        i = 0
        for frame in container.decode(video=0):
            if i > max_inds + 1:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['imgs'] = [imgs[i % len(imgs)] for i in results['frame_inds']]

        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread})'
        return repr_str


@PIPELINES.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    """Using pyav to decode the motion vectors from video.

    Reference: https://github.com/PyAV-Org/PyAV/
        blob/main/tests/test_decode.py

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "motion_vectors", "frame_inds".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        """Parse the returned vectors."""
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def __call__(self, results):
        """Perform the PyAV motion vector decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max index to make early stop
        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:
                    # Key frame don't have motion vectors
                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        frame_dict = {
            idx: container[idx].asnumpy()
            for idx in np.unique(frame_inds)
        }

        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class OpenCVInit:
    """Using OpenCV to initialize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@PIPELINES.register_module()
class OpenCVDecode:
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


def kittidepth_depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    #depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    #assert(np.max(depth_png) > 255)
    depth_png = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32')
    #print("d1 >> ", depth_png[200:300,500:700])
    #print("d2>> ", d2[200:300,500:700])
    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1
    #print("d2>> ", depth[200:300,500:700])
    return depth

def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img
def load_depth_midair(z_path):
    depth = open_float16(z_path)
    w, h = depth.shape
    #print("depth shape >>.", depth.shape)
    f = w/2
    # tranfrom to camera coord
    x_i, y_i = np.meshgrid(
                                np.arange(0,w,1),
                                np.arange(0,h,1), indexing="xy")
    #print("xi >> \n", x_i)
    #print("y_i >> \n ", y_i)
    
    r_xy = depth / np.sqrt((x_i-(h/2))**2 + (y_i-(h/2))**2 + f**2)
    #x_c = r_xy*(x_i-h/2)
    #y_c = r_xy*(y_i-h/2)
    z_c = r_xy*(f)
    z_c[depth>=65500.0] = 65500.0
    #print("z_c >> \n ", z_c)
    return z_c #np.expand_dims(z_c, axis=0)
    

@PIPELINES.register_module()
class StereoRawFrameDecode:
    """Load and decode frames with given indices. For the stereo data sequences

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "left_imgs", "right_imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, stereo_id=['image_02', 'image_03'],dataset=None, depth_scale=None, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.stereo_id = stereo_id
        self.depth_scale = depth_scale
        self.dataset = dataset
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        if "depth_dir" in results:
            depth_directory = results['depth_dir']
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        left_imgs = list()
        right_imgs = list()
        left_depths = list()
        right_depths = list()
        left_masks = list()
        left_pred_depths = list()
        right_pred_depths = list()
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath_left = osp.join(directory, self.stereo_id[0], filename_tmpl.format(frame_idx))
                left_img_bytes = self.file_client.get(filepath_left)
                # Get frame with channel order RGB directly.
                left_cur_frame = mmcv.imfrombytes(left_img_bytes, channel_order='rgb')
                left_imgs.append(left_cur_frame)
                # right images saving
                #if len(self.stereo_id) ==2:
                filepath_right = osp.join(directory, self.stereo_id[1], filename_tmpl.format(frame_idx))
                right_img_bytes = self.file_client.get(filepath_right)
                # Get frame with channel order RGB directly.
                right_cur_frame = mmcv.imfrombytes(right_img_bytes, channel_order='rgb')
                right_imgs.append(right_cur_frame)
                #if len(self.stereo_id)==4:
                if "pred_depth_dir_left" in results:
                    # left depth
                    filepath_left_depth = osp.join(results['pred_depth_dir_left'], filename_tmpl.format(frame_idx))
                    left_depth_cur_frame = 0.01 * cv2.imread(filepath_left_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    left_pred_depths.append(left_depth_cur_frame)
                if "pred_depth_dir_right" in results:
                    filepath_right_depth = osp.join(results['pred_depth_dir_right'], filename_tmpl.format(frame_idx))
                    right_depth_cur_frame = 0.01 * cv2.imread(filepath_right_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    right_pred_depths.append(right_depth_cur_frame)
                if "mask_filename_tmpl" in results:
                    if self.dataset == "midair":
                        mask_filename_tmpl = results['mask_filename_tmpl']
                        filepath_left_depth = osp.join(depth_directory, self.stereo_id[3], mask_filename_tmpl.format(frame_idx))
                        left_gt_occlu_mask_cur_frame = cv2.imread(filepath_left_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        left_gt_occlu_mask_cur_frame[left_gt_occlu_mask_cur_frame==255] = 1
                        left_masks.append(left_gt_occlu_mask_cur_frame)
                if "depth_filename_tmpl" in results:
                    assert self.dataset is not None, "to load GT depth, you should give dataset type  "
                    if self.depth_scale is not None:
                        warnings.warn('"depth sacle" is deprecated, you only need to give dataset type in "StereoRawFrameDecode" cfg', \
                            DeprecationWarning)
                    #if "depth_dir" in results:
                    depth_filename_tmpl = results['depth_filename_tmpl']
                    # left depth
                    filepath_left_depth = osp.join(depth_directory, self.stereo_id[2], depth_filename_tmpl.format(frame_idx))
                    if self.dataset =="vkitti2":
                        left_depth_cur_frame = 0.01 * cv2.imread(filepath_left_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                        #left_depth_cur_frame[left_depth_cur_frame>=655] = 65555
                    elif self.dataset =="kittidepth":
                        left_depth_cur_frame = kittidepth_depth_read(filepath_left_depth)
                    elif self.dataset == "midair":
                        left_depth_cur_frame = load_depth_midair(filepath_left_depth)
                    else:
                        raise ValueError
                    # cm ->m 
                    # TODO: make the depth scale  dealed in dataset class
                    #left_depth_cur_frame = left_depth_cur_frame * self.depth_scale
                    left_depths.append(left_depth_cur_frame)
                    # right depth
                    if self.dataset == "midair":
                        continue
                    else:
                        filepath_right_depth = osp.join(depth_directory, self.stereo_id[3], depth_filename_tmpl.format(frame_idx)) 
                    
                    if self.dataset =="vkitti2":
                        # centmeter -> meter
                        right_depth_cur_frame = 0.01* cv2.imread(filepath_right_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                        #right_depth_cur_frame[right_depth_cur_frame>=655] = 65555
                    elif self.dataset =="kittidepth":
                        right_depth_cur_frame = kittidepth_depth_read(filepath_right_depth)
                    elif self.dataset =="midair":
                        right_depth_cur_frame = None # they didn't provide GT right depth directly.
                    else:
                        raise ValueError
                    #right_depth_cur_frame = right_depth_cur_frame * self.depth_scale
                    right_depths.append(right_depth_cur_frame)

            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                left_imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['left_imgs'] = left_imgs
        #if len(self.stereo_id) ==2:
        results['right_imgs'] = right_imgs
        if "pred_depth_dir_left" in results:
            results['left_pred_depths'] = left_pred_depths
        if "pred_depth_dir_right" in results:
            results['right_pred_depths'] = right_pred_depths
        if "depth_dir" in results:
            results['left_depths'] = left_depths
            if self.dataset == "midair":
                results['right_depths'] = None
            else:
                results['right_depths'] = right_depths
        if "mask_filename_tmpl" in results:
            results['left_masks'] = left_masks
        results['original_shape'] = left_imgs[0].shape[:2]
        results['img_shape'] = left_imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        #if 'gt_bboxes' in results:
        #    h, w = results['img_shape']
        #    scale_factor = np.array([w, h, w, h])
        #    gt_bboxes = results['gt_bboxes']
        #    gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
        #    results['gt_bboxes'] = gt_bboxes
        #    if 'proposals' in results and results['proposals'] is not None:
        #        proposals = results['proposals']
        #        proposals = (proposals * scale_factor).astype(np.float32)
        #        results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

def read_disparity_cityspaces(path):
    p = np.array(Image.open(path), dtype=np.float16)
    #print("p>> ", p)
    d = ( p - 1. ) / 256
    d[p<=1] = 1e-8
    #print("load disparity >> ", d[500:550,1000:1050])
    return d

def read_disparity_kitti_stereo_matching(path):
    disp = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
    return disp



@PIPELINES.register_module()
class StereoDefinedFrameDecode:
    """Load and decode frames with given indices. For the stereo data sequences

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "left_imgs", "right_imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, dataset="kittidepth", io_backend='disk', decoding_backend='cv2', channel_order="rgb",  **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        #self.stereo_id = stereo_id
        #self.depth_scale = depth_scale
        self.channel_order = channel_order
        self.dataset = dataset
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        if "depth_dir" in results:
            depth_directory = results['depth_dir']
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, )#**self.kwargs)

        left_imgs = list()
        right_imgs = list()
        left_depths = list()
        right_depths = list()
        left_disps = list()
        right_disps = list()
        #if results['frame_inds'].ndim != 1:
        #    results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        #print(results["left_frame_paths"])
        #for i in range(len(results["left_frame_paths"])):
        if modality == 'RGB':
            for i, filepath_left in enumerate(results["left_frame_paths"]):
                left_img_bytes = self.file_client.get(filepath_left)
                # Get frame with channel order RGB directly.
                left_cur_frame = mmcv.imfrombytes(left_img_bytes, channel_order=self.channel_order)
                left_imgs.append(left_cur_frame)
            # right images saving
            #if len(self.stereo_id) ==2:
            for i, filepath_right in enumerate(results["right_frame_paths"]):
                right_img_bytes = self.file_client.get(filepath_right)
                # Get frame with channel order RGB directly.
                right_cur_frame = mmcv.imfrombytes(right_img_bytes, channel_order=self.channel_order)
                right_imgs.append(right_cur_frame)
            
            if "left_depth_paths" in results:
                #depth_filename_tmpl = results['depth_filename_tmpl']
                # left depth
                for i, filepath_left_depth in enumerate(results["left_depth_paths"]):
                    if self.dataset =="vkitti2":
                        left_depth_cur_frame = 0.01* cv2.imread(filepath_left_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    elif self.dataset =="kittidepth":
                        left_depth_cur_frame = kittidepth_depth_read(filepath_left_depth)
                    elif self.dataset == "cityspaces":
                        left_depth_cur_frame = results["focal"]*results["baseline"] / read_disparity_cityspaces(filepath_left_depth)
                    
                    else:
                        raise ValueError                
                    # cm ->m 
                    # TODO: make the depth scale  dealed in dataset class
                    #left_depth_cur_frame = left_depth_cur_frame * self.depth_scale
                    left_depths.append(left_depth_cur_frame)
            if "right_depth_paths" in results:
                # right depth
                for i, filepath_right_depth in enumerate(results["right_depth_paths"]):
                    if self.dataset =="vkitti2":
                        right_depth_cur_frame = 0.01* cv2.imread(filepath_right_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    elif self.dataset =="kittidepth":
                        right_depth_cur_frame = kittidepth_depth_read(filepath_right_depth)  
                    elif self.dataset == "cityspaces":
                        right_depth_cur_frame = results["focal"]*results["baseline"] / read_disparity_cityspaces(filepath_right_depth)

                    else:
                        raise ValueError                
                    #right_depth_cur_frame = right_depth_cur_frame * self.depth_scale
                    right_depths.append(right_depth_cur_frame)

            if "left_disp_paths" in results:
                for i, filepath_left_disp in enumerate(results["left_disp_paths"]):
                    if self.dataset=="sceneflow":
                        from ..dataset_utils import load_scene_flow_disp
                        disp = load_scene_flow_disp(filepath_left_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset=="crestereo":
                        from ..dataset_utils import load_crestereo_disp
                        disp = load_crestereo_disp(filepath_left_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset=="crestereo_sceneflow":
                        from ..dataset_utils import load_scene_flow_disp
                        from ..dataset_utils import load_crestereo_disp
                        if filepath_left_disp.split('.')[-1] == "png":
                            disp = load_crestereo_disp(filepath_left_disp)
                        elif filepath_left_disp.split('.')[-1] == "pfm":
                            disp = load_scene_flow_disp(filepath_left_disp)
                        else:
                            raise ValueError
                    elif self.dataset=="middlebury":
                        from ..dataset_utils import load_middlebury_disp
                        disp = load_middlebury_disp(filepath_left_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset == "kittistereomatching":
                        disp = read_disparity_kitti_stereo_matching(filepath_left_disp)
                        #disp[disp==0] = 1e-3
                        #left_depth_cur_frame = results["focal"]*results["baseline"] / disp
                    else:
                        raise ValueError           
                    left_disps.append(disp)
            if "right_disp_paths" in results:
                for i, filepath_right_disp in enumerate(results["right_disp_paths"]):
                    if self.dataset=="sceneflow":
                        from ..dataset_utils import load_scene_flow_disp
                        disp = load_scene_flow_disp(filepath_right_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset=="crestereo":
                        from ..dataset_utils import load_crestereo_disp
                        disp = load_crestereo_disp(filepath_right_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset=="crestereo_sceneflow":
                        from ..dataset_utils import load_scene_flow_disp
                        from ..dataset_utils import load_crestereo_disp
                        if filepath_left_disp.split('.')[-1] == "png":
                            disp = load_crestereo_disp(filepath_left_disp)
                        elif filepath_left_disp.split('.')[-1] == "pfm":
                            disp = load_scene_flow_disp(filepath_left_disp)
                        else:
                            raise ValueError
                    elif self.dataset =="middlebury":
                        from ..dataset_utils import load_middlebury_disp
                        disp = load_middlebury_disp(filepath_right_disp)
                        assert len(disp.shape) ==2
                    elif self.dataset == "kittistereomatching":
                        disp = read_disparity_kitti_stereo_matching(filepath_right_disp)
                        #disp[disp==0] = 1e-3
                        #right_depth_cur_frame = results["focal"]*results["baseline"] / disp
                    else:
                        raise ValueError      
                    right_disps.append(disp)     
        else:
            raise NotImplementedError

        results['left_imgs'] = left_imgs
        #if len(self.stereo_id) ==2:
        results['right_imgs'] = right_imgs
        if "left_depth_paths" in results:
            results['left_depths'] = left_depths
        if "right_depth_paths" in results:
            results['right_depths'] = right_depths
        if "left_disp_paths" in results:
            results["left_disps"] = left_disps
        if "right_disp_paths" in results:
            results["right_disps"] = right_disps
        results['original_shape'] = left_imgs[0].shape[:2]
        results['img_shape'] = left_imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        #if 'gt_bboxes' in results:
        #    h, w = results['img_shape']
        #    scale_factor = np.array([w, h, w, h])
        #    gt_bboxes = results['gt_bboxes']
        #    gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
        #    results['gt_bboxes'] = gt_bboxes
        #    if 'proposals' in results and results['proposals'] is not None:
        #        proposals = results['proposals']
        #        proposals = (proposals * scale_factor).astype(np.float32)
        #        results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

@PIPELINES.register_module()
class VOFrameDecode:
    """Load and decode frames with given indices.

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', channel_order="rgb", **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            if modality == 'RGB':
                filepath = osp.join(directory, filename_tmpl.format(frame_idx))
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order=self.channel_order)
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.extend([x_frame, y_frame])
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

@PIPELINES.register_module()
class ImageDecode:
    """Load and decode images.

    Required key is "filename", added or modified keys are "imgs", "img_shape"
    and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``ImageDecode`` to load image given the file path.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results

 

 

@PIPELINES.register_module()
class BuildPseudoClip:
    """Build pseudo clips with one single image by repeating it n times.

    Required key is "imgs", added or modified key is "imgs", "num_clips",
        "clip_len".

    Args:
        clip_len (int): Frames of the generated pseudo clips.
    """

    def __init__(self, clip_len):
        self.clip_len = clip_len

    def __call__(self, results):
        # the input should be one single image
        assert len(results['imgs']) == 1
        im = results['imgs'][0]
        for _ in range(1, self.clip_len):
            results['imgs'].append(np.copy(im))
        results['clip_len'] = self.clip_len
        results['num_clips'] = 1
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@PIPELINES.register_module()
class FrameSelector(RawFrameDecode):
    """Deprecated class for ``RawFrameDecode``."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"FrameSelector" is deprecated, please switch to'
                      '"RawFrameDecode"')
        super().__init__(*args, **kwargs)



@PIPELINES.register_module()
class LoadStereoImages(object):
    """Load an image from file.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                views=["left", "right"],
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2',
                 to_rgb=False,
                 to_gray=False):
        self.views = views
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.to_rgb = to_rgb
        self.to_gray = to_gray

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        for v in self.views:
            filenames = results[f'{v}_frame_paths']
            imgs = []
            for f in filenames:
                img_bytes = self.file_client.get(f)
                channel_order= 'rgb' if self.to_rgb else 'bgr'
                img = mmcv.imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.imdecode_backend, channel_order=channel_order)
                if self.to_float32:
                    img = img.astype(np.float32)
                if self.to_gray:
                    if channel_order == 'rgb':
                        img = mmcv.rgb2gray(img).reshape(img.shape[:2]+(1,))
                    elif channel_order == 'bgr':
                        img = mmcv.bgr2gray(img).reshape(img.shape[:2]+(1,))
                    else:
                        raise ValueError(f'Invalid channel order {channel_order}.')
                imgs.append(img)
            results[f"{v}_imgs"] = imgs
            results['img_shape'] = imgs[0].shape[:2]
            results['ori_shape'] = imgs[0].shape[:2]
            results['pad_shape'] = imgs[0].shape[:2]
            results['scale_factor'] = 1.0
            num_channels = 1 if len(img.shape) < 3 else img.shape[2] # h w 3. default BGR
            results['img_norm_cfg'] = dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=self.to_rgb)
    
        return results


@PIPELINES.register_module()
class LoadAnnotations(object):
    def __init__(self,
                 views=[],
                 modalities=[],
                 ):
        self.views = views
        self.modalities = modalities
        assert len(views) > 0 and len(modalities)>0, "give keys if use this class, left_depth, right_depth, left_disp, right_disp "
         
 
    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`Dataset`.
        Returns:
            dict: The dict contains loaded depth or disparity annotations.
        """
 
         
        for v in self.views:
            for m in self.modalities:
                
                sequential_gt_labels = []
                # transfrom mobility 
                if not (f'{v}_{m}_paths' in results and results[f'{v}_{m}_paths'] is not None):
                    m2 = "disp" if m == "depth" else "depth"
                    filenames = results[f'{v}_{m2}_paths']
                    focal, baseline = results["focal"], results["baseline"]
                    for i, filepath in enumerate(filenames):
                        if filepath.endswith('.png') or filepath.endswith('.jpg'):
                            assert "depth_scale_ratio" in results
                            gt_label = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32) / results["depth_scale_ratio"]
                        elif filepath.endswith('.pfm'):
                            from ..dataset_utils import load_pfm
                            gt_label, sca = load_pfm(filepath)
                        else:
                            raise ValueError(f"unknown file type {filepath}")
                        sparse_indx = (gt_label > 0) # only compute valid depths values
                        #print("depth map", gt_label[sparse_indx])
                        gt_label[sparse_indx] = focal*baseline / (gt_label[sparse_indx])
                        #gt_label = focal*baseline / (gt_label+1e-6)
                        #print("trans disp maps", gt_label[sparse_indx])
                        assert len(gt_label.shape) ==2
                        sequential_gt_labels.append(gt_label)
                else: # directly load 
                    filenames = results[f'{v}_{m}_paths']
                    for i, filepath in enumerate(filenames):
                        if filepath.endswith('.png') or filepath.endswith('.jpg'):
                            assert "depth_scale_ratio" in results
                            gt_label = cv2.imread(filepath, cv2.IMREAD_UNCHANGED).astype(np.float32) / results["depth_scale_ratio"]
                        elif filepath.endswith('.pfm'):
                            from ..dataset_utils import load_pfm
                            gt_label, sca = load_pfm(filepath)
                        else:
                            raise ValueError(f"unknown file type {filepath}")
                        assert len(gt_label.shape) ==2
                        sequential_gt_labels.append(gt_label)
                
                results[f"{v}_{m}s"] = sequential_gt_labels
        return results