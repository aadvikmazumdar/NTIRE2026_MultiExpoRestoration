import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import glob
import random


class RawFusionDataset(data.Dataset):
    """NTIRE 2026 Burst HDR dataset.
    
    Dataset structure (all flat, no subdirs):
      trn/Scene-000-gt.tif, Scene-000-in-0.tif ... Scene-000-in-8.tif
      val_input/Scene-300-in-0.tif ... Scene-300-in-8.tif
      val_gt/Scene-300-gt.tif
    
    Input frames: 1ch RAW 16-bit, 768x1536
    GT: 3ch RGB 16-bit, 768x1536
    """
    def __init__(self, input_dir, gt_dir=None, crop_size=None, augment=False):
        super().__init__()
        self.input_dir = input_dir
        self.gt_dir = gt_dir if gt_dir else input_dir
        self.crop_size = crop_size
        self.augment = augment

        # find all scene indices from input files
        in_files = sorted(glob.glob(os.path.join(input_dir, 'Scene-*-in-0.tif')))
        self.scene_ids = []
        for f in in_files:
            name = os.path.basename(f)
            idx = name.split('-')[1]  # "000", "001", etc
            self.scene_ids.append(idx)
        print(f"Found {len(self.scene_ids)} scenes in {input_dir}")

    def __len__(self):
        return len(self.scene_ids)

    def _read_raw(self, path):
        """Read single-channel RAW .tif (float32 [0,1]) -> tensor [1, H, W]"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if img.ndim == 3:
            img = img[:, :, 0]
        return torch.from_numpy(img).unsqueeze(0)

    def _read_rgb(self, path):
        """Read 3-channel RGB .tif (float32 [0,1]) -> tensor [3, H, W]"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1)

    def _random_crop(self, frames, gt):
        """Random crop maintaining 2:1 aspect ratio (matching full res)."""
        _, H, W = gt.shape
        ch, cw = self.crop_size
        if H < ch or W < cw:
            return frames, gt
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        frames = frames[:, top:top+ch, left:left+cw]
        gt = gt[:, top:top+ch, left:left+cw]
        return frames, gt

    def _augment(self, frames, gt):
        """Random flips (horizontal + vertical)."""
        if random.random() > 0.5:
            frames = torch.flip(frames, [-1])
            gt = torch.flip(gt, [-1])
        if random.random() > 0.5:
            frames = torch.flip(frames, [-2])
            gt = torch.flip(gt, [-2])
        return frames, gt

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]

        # load 9 input frames -> [9, H, W]
        input_frames = []
        for i in range(9):
            path = os.path.join(self.input_dir, f'Scene-{sid}-in-{i}.tif')
            input_frames.append(self._read_raw(path))
        frames = torch.cat(input_frames, dim=0)  # [9, H, W]

        # load GT -> [3, H, W]
        gt_path = os.path.join(self.gt_dir, f'Scene-{sid}-gt.tif')
        if os.path.exists(gt_path):
            gt = self._read_rgb(gt_path)
        else:
            gt = torch.zeros(3, frames.shape[1], frames.shape[2])

        if self.crop_size is not None:
            frames, gt = self._random_crop(frames, gt)
        if self.augment:
            frames, gt = self._augment(frames, gt)

        return frames, gt


class RawFusionTestDataset(data.Dataset):
    """Test dataset - input only, no GT."""
    def __init__(self, input_dir):
        super().__init__()
        self.input_dir = input_dir
        in_files = sorted(glob.glob(os.path.join(input_dir, 'Scene-*-in-0.tif')))
        self.scene_ids = []
        for f in in_files:
            name = os.path.basename(f)
            idx = name.split('-')[1]
            self.scene_ids.append(idx)
        print(f"Found {len(self.scene_ids)} test scenes in {input_dir}")

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, idx):
        sid = self.scene_ids[idx]
        input_frames = []
        for i in range(9):
            path = os.path.join(self.input_dir, f'Scene-{sid}-in-{i}.tif')
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 3:
                img = img[:, :, 0]
            t = torch.from_numpy(img).unsqueeze(0)
            input_frames.append(t)
        frames = torch.cat(input_frames, dim=0)
        return frames, f'Scene-{sid}'
