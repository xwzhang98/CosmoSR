import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from ..utils import import_attr
from . import norms


class FieldDataset(Dataset):
    """Paired periodic displacement/velocity crops with cosmological styles.

    Files are sorted within each pattern; the launcher first validates paired
    simulation IDs. Axis flips, permutations and shifts augment vector fields.
    HR crop dimensions are scaled relative to LR; padding is in lattice cells.
    """

    def __init__(
        self,
        in_patterns,
        tgt_patterns,
        style_pattern=None,
        in_norms=None,
        tgt_norms=None,
        callback_at=None,
        augment=False,
        aug_shift=None,
        crop=None,
        crop_start=None,
        crop_stop=None,
        crop_step=None,
        in_pad=0,
        tgt_pad=0,
        scale_factor=1,
        **kwargs,
    ):
        in_file_lists = [sorted(glob(p)) for p in in_patterns]
        num_snapshots = len(in_file_lists[0])
        # Preserve the historical RNG draw before model initialization.
        list(
            WeightedRandomSampler(
                weights=torch.ones(num_snapshots),
                num_samples=min(16, num_snapshots),
                replacement=False,
            )
        )
        self.in_files = list(zip(*in_file_lists))
        tgt_file_lists = [sorted(glob(p)) for p in tgt_patterns]
        self.tgt_files = list(zip(*tgt_file_lists))
        if len(self.in_files) != len(self.tgt_files):
            raise ValueError("number of input and target fields do not match")
        self.nfile = len(self.in_files)
        if self.nfile == 0:
            raise FileNotFoundError("file not found for {}".format(in_patterns))
        self.is_read_once = np.full(self.nfile, False)
        self.in_chan = [np.load(f, mmap_mode="r").shape[0] for f in self.in_files[0]]
        self.tgt_chan = [np.load(f, mmap_mode="r").shape[0] for f in self.tgt_files[0]]
        self.size = np.load(self.in_files[0][0], mmap_mode="r").shape[1:]
        self.size = np.asarray(self.size)
        self.ndim = len(self.size)
        self.style = style_pattern is not None
        self.style_size = 0
        if self.style:
            style_files = sorted(glob(style_pattern))
            self.style_files = style_files
            if len(self.style_files) != len(self.in_files):
                raise ValueError("number of style and input files do not match")
            self.style_size = np.load(self.style_files[0]).shape[0]
        if in_norms is not None and len(in_patterns) != len(in_norms):
            raise ValueError(
                "numbers of input normalization functions and fields do not match"
            )
        self.in_norms = in_norms
        if tgt_norms is not None and len(tgt_patterns) != len(tgt_norms):
            raise ValueError(
                "numbers of target normalization functions and fields do not match"
            )
        self.tgt_norms = tgt_norms
        self.callback_at = callback_at
        self.augment = augment
        if self.ndim == 1 and self.augment:
            raise ValueError("cannot augment 1D fields")
        self.aug_shift = np.broadcast_to(aug_shift, (self.ndim,))
        if crop is None:
            self.crop = self.size
        else:
            self.crop = np.broadcast_to(crop, (self.ndim,))
        if crop_start is None:
            crop_start = np.zeros_like(self.size)
        else:
            crop_start = np.broadcast_to(crop_start, (self.ndim,))
        if crop_stop is None:
            crop_stop = self.size
        else:
            crop_stop = np.broadcast_to(crop_stop, (self.ndim,))
        if crop_step is None:
            crop_step = self.crop
        else:
            crop_step = np.broadcast_to(crop_step, (self.ndim,))
        self.crop_step = crop_step
        self.anchors = np.stack(
            np.mgrid[
                tuple(
                    (
                        slice(crop_start[d], crop_stop[d], crop_step[d])
                        for d in range(self.ndim)
                    )
                )
            ],
            axis=-1,
        ).reshape(-1, self.ndim)
        self.ncrop = len(self.anchors)

        def format_pad(pad, ndim):
            if isinstance(pad, int):
                pad = np.broadcast_to(pad, ndim * 2)
            elif isinstance(pad, tuple) and len(pad) == ndim:
                pad = np.repeat(pad, 2)
            elif isinstance(pad, tuple) and len(pad) == ndim * 2:
                pad = np.array(pad)
            else:
                raise ValueError("pad and ndim mismatch")
            return pad.reshape(ndim, 2)

        self.in_pad = format_pad(in_pad, self.ndim)
        self.tgt_pad = format_pad(tgt_pad, self.ndim)
        if scale_factor != 1:
            tgt_size = np.load(self.tgt_files[0][0], mmap_mode="r").shape[1:]
            if any(self.size * scale_factor != tgt_size):
                raise ValueError("input size x scale factor != target size")
        self.scale_factor = scale_factor
        self.nsample = self.nfile * self.ncrop
        self.kwargs = kwargs
        self.commonpath = os.path.commonpath(
            file for files in self.in_files[:2] + self.tgt_files[:2] for file in files
        )

    def __len__(self):
        return self.nsample

    def __getitem__(self, idx):
        (ifile, icrop) = divmod(idx, self.ncrop)
        if self.is_read_once[ifile]:
            mmap_mode = "r"
        else:
            mmap_mode = None
            self.is_read_once[ifile] = True
        in_fields = [np.load(f, mmap_mode="r") for f in self.in_files[ifile]]
        tgt_fields = [np.load(f, mmap_mode="r") for f in self.tgt_files[ifile]]
        anchor = self.anchors[icrop]
        for d, shift in enumerate(self.aug_shift):
            if shift is not None:
                anchor[d] += torch.randint(int(shift), (1,))
        if self.augment:
            perm_axes = perm([], None, self.ndim)
            argsort_perm_axes = np.argsort(perm_axes.numpy())
        else:
            argsort_perm_axes = slice(None)
        crop(
            in_fields,
            anchor,
            self.crop[argsort_perm_axes],
            self.in_pad[argsort_perm_axes],
        )
        crop(
            tgt_fields,
            anchor * self.scale_factor,
            self.crop[argsort_perm_axes] * self.scale_factor,
            self.tgt_pad[argsort_perm_axes],
        )
        in_fields = [
            torch.from_numpy(np.asarray(f, dtype=np.float32)) for f in in_fields
        ]
        tgt_fields = [
            torch.from_numpy(np.asarray(f, dtype=np.float32)) for f in tgt_fields
        ]
        style = torch.empty(0, dtype=torch.float32)
        if self.style:
            style_np = np.load(self.style_files[ifile], mmap_mode="r")
            style = torch.from_numpy(np.array(style_np, dtype=np.float32, copy=True))
        if self.in_norms is not None:
            for norm, x in zip(self.in_norms, in_fields):
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                norm(x, a=style[0], Om=style[1], **self.kwargs)
        if self.tgt_norms is not None:
            for norm, x in zip(self.tgt_norms, tgt_fields):
                norm = import_attr(norm, norms, callback_at=self.callback_at)
                norm(x, a=style[0], Om=style[1], **self.kwargs)
        if self.augment:
            flip_axes = flip(in_fields, None, self.ndim)
            flip_axes = flip(tgt_fields, flip_axes, self.ndim)
            perm_axes = perm(in_fields, perm_axes, self.ndim)
            perm_axes = perm(tgt_fields, perm_axes, self.ndim)
        in_fields = torch.cat(in_fields, dim=0)
        tgt_fields = torch.cat(tgt_fields, dim=0)
        tgt_relpath = [
            os.path.relpath(file, start=self.commonpath)
            for file in self.tgt_files[ifile]
        ]
        return {
            "input": in_fields,
            "target": tgt_fields,
            "style": style,
            "target_relpath": tgt_relpath,
        }


def crop(fields, anchor, crop, pad):
    """Faster periodic crop: applies 1-D wrap selection per axis sequentially.
    Avoids broadcasting gigantic index grids."""
    ref = fields[0]
    size = ref.shape[1:]
    ndim = len(size)
    if any((x.shape[1:] != size for x in fields[1:])):
        raise RuntimeError(f"shape mismatch: {[x.shape[1:] for x in fields]}")
    if not ndim == len(anchor) == len(crop) == len(pad):
        raise RuntimeError("ndim mismatch")
    idxs = []
    for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, size)):
        start = int(a) - int(p0)
        length = int(c) + int(p0) + int(p1)
        idx = np.arange(start, start + length, dtype=np.int64) % int(s)
        idxs.append(idx)
    for i, x in enumerate(fields):
        for d, idx in enumerate(idxs):
            x = np.take(x, idx, axis=1 + d, mode="wrap")
        fields[i] = x
    return tuple((slice(None) for _ in range(ndim + 1)))


def flip(fields, axes, ndim):
    if ndim == 1:
        raise RuntimeError("flipping is ambiguous for 1D scalars/vectors")
    if axes is None:
        axes = torch.randint(2, (ndim,), dtype=torch.bool)
        axes = torch.arange(ndim)[axes]
    for i, x in enumerate(fields):
        if x.shape[0] == ndim:
            x[axes] = -x[axes]
        shifted_axes = (1 + axes).tolist()
        x = torch.flip(x, shifted_axes)
        fields[i] = x
    return axes


def perm(fields, axes, ndim):
    if ndim == 1:
        raise RuntimeError("permutation is not necessary for 1D fields")
    if axes is None:
        axes = torch.randperm(ndim)
    for i, x in enumerate(fields):
        if x.shape[0] == ndim:
            x = x[axes]
        shifted_axes = [0] + (1 + axes).tolist()
        x = x.permute(shifted_axes)
        fields[i] = x
    return axes
