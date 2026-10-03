from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from monai.data.meta_obj import get_track_meta
from monai.transforms import MapTransform, RandomizableTransform
from monai.transforms.utils import convert_to_tensor


class RandCropandZero(RandomizableTransform):
    """
    Randomly crops and then zero-pads a 3D image tensor, optionally per channel.

    This transform randomly selects crop sizes for each spatial dimension (depth, height, width)
    up to a maximum value, removes those voxels from the borders, and then pads the cropped image
    back to its original size with the minimum value of the input tensor (effectively zeroing out
    the cropped regions). The cropping and padding can be applied independently to each channel.

    Attributes:
        max_crop (int): Maximum number of voxels to crop from each side of each spatial dimension.
        prob (float): Probability of applying the transform.
        channel_wise (bool): If True, apply random cropping independently to each channel.
        crop_offsets (Optional[List[List[Tuple[int, int]]]]): Stores the crop offsets for each channel and dimension.

    Methods:
        randomize(img: Any = None) -> None:
            Randomly determines crop offsets for each spatial dimension and channel (if enabled).

        apply_with_stored_offsets(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            Applies cropping and zero-padding to the input image using the previously determined offsets.

        __call__(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            Randomizes crop offsets and applies the transform to the input image.

    Args:
        max_crop (int, optional): Maximum crop size per side per dimension. Defaults to 20.
        prob (float, optional): Probability of applying the transform. Defaults to 0.1.
        channel_wise (bool, optional): Whether to apply cropping independently per channel. Defaults to False.

    Example:
        >>> transform = RandCropandZero(max_crop=10, prob=0.5, channel_wise=True)
        >>> out = transform(input_tensor)
    """

    def __init__(
        self, max_crop: int = 20, prob: float = 0.1, channel_wise: bool = False
    ):
        super().__init__(prob)
        self.max_crop = max_crop
        self.channel_wise = channel_wise
        self.crop_offsets = None

    def randomize(self, img: Any = None) -> None:
        super().randomize(img)
        if not self._do_transform:
            self.crop_offsets = None
            return

        shape = img.shape[-3:]
        num_channels = img.shape[0] if self.channel_wise else 1

        self.crop_offsets = []
        for c in range(num_channels):
            offsets = []
            for dim_size in shape:
                crop_before = self.R.randint(0, self.max_crop + 1)
                crop_after = self.R.randint(0, self.max_crop + 1)
                offsets.append((crop_before, crop_after))
            self.crop_offsets.append(offsets)

    def apply_with_stored_offsets(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if not self._do_transform or self.crop_offsets is None:
            return img

        spatial_dims = img.shape[-3:]
        non_spatial_slices = [slice(None)] * (img.ndim - 3)

        pad_value = torch.min(img)

        if self.channel_wise:
            out = []
            for c in range(img.shape[0]):
                slices = [
                    slice(
                        self.crop_offsets[c][i][0],
                        spatial_dims[i] - self.crop_offsets[c][i][1],
                    )
                    for i in range(3)
                ]
                cropped = img[c, *slices]
                pad_width = [self.crop_offsets[c][i] for i in range(3)]
                padded = F.pad(
                    cropped,
                    [p for pair in reversed(pad_width) for p in pair],
                    mode="constant",
                    value=pad_value,
                )
                out.append(padded)
            return torch.stack(out, dim=0)
        else:
            slices = [
                slice(
                    self.crop_offsets[0][i][0],
                    spatial_dims[i] - self.crop_offsets[0][i][1],
                )
                for i in range(3)
            ]
            cropped = img[*non_spatial_slices, *slices]
            pad_width = [self.crop_offsets[0][i] for i in range(3)]
            padded = F.pad(
                cropped,
                [p for pair in reversed(pad_width) for p in pair],
                mode="constant",
                value=pad_value,
            )
            return padded

    def __call__(self, img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        self.randomize(img)
        return self.apply_with_stored_offsets(img)


class RandCropandZerod(MapTransform, RandomizableTransform):
    """
    Dictionary-based random crop and zeroing transform for multi-key data.
    This transform randomly crops and zeroes out regions of input arrays for the specified keys.
    The same random crop offsets are applied to all keys to ensure spatial consistency.
    It supports optional channel-wise cropping and can be applied probabilistically.
    Args:
        keys (Sequence[str]):
            Keys of the corresponding items to be transformed.
        max_crop (int, optional):
            Maximum number of voxels/pixels to crop from each side. Defaults to 20.
        prob (float, optional):
            Probability of applying the transform. Defaults to 0.1.
        channel_wise (bool, optional):
            If True, applies cropping independently for each channel. Defaults to False.
        allow_missing_keys (bool, optional):
            If True, skips keys that are missing in the input dictionary. Defaults to False.
    Attributes:
        max_crop (int):
            Maximum crop size.
        channel_wise (bool):
            Whether to apply cropping per channel.
        _transform (RandCropandZero):
            Internal transform instance used to perform the actual cropping and zeroing.
    Methods:
        set_random_state(seed=None, state=None):
            Sets the random state for reproducibility.
        randomize_offsets(reference):
            Randomizes crop offsets based on a reference image.
        __call__(data):
            Applies the transform to the input dictionary, using consistent offsets for all keys.
    Example:
        >>> transform = RandCropandZerod(keys=["image", "label"], max_crop=10, prob=0.2)
        >>> output = transform({"image": img_array, "label": label_array})
    """

    def __init__(
        self,
        keys,
        max_crop: int = 20,
        prob: float = 0.1,
        channel_wise: bool = False,
        allow_missing_keys=False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.max_crop = max_crop
        self.channel_wise = channel_wise
        self._transform = RandCropandZero(
            max_crop=max_crop, prob=1.0, channel_wise=channel_wise
        )

    def set_random_state(self, seed=None, state=None):
        super().set_random_state(seed, state)
        self._transform.set_random_state(seed, state)
        return self

    def randomize_offsets(self, reference):
        """Randomize crop offsets once based on a reference image (usually the first key)"""
        self._transform.randomize(reference)

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return {
                k: convert_to_tensor(v, track_meta=get_track_meta())
                for k, v in d.items()
            }

        # Pick the first key as reference to generate consistent offsets
        first_key = self.first_key(d)
        if first_key == ():
            return {
                k: convert_to_tensor(v, track_meta=get_track_meta())
                for k, v in d.items()
            }

        # Randomize offsets once for all keys
        self.randomize_offsets(d[first_key])

        # Apply the same offsets to all keys
        for key in self.key_iterator(d):
            d[key] = self._transform.apply_with_stored_offsets(
                d[key]
            )  # apply stored offsets
        return d
