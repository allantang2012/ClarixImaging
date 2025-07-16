import random
import torch
from PIL import Image
from glob import glob


class Places2(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # use about 8M images in the challenge dataset
        if split == 'train':
            # Try both .jpg and .JPEG extensions
            self.paths = glob('{:s}/**/*.jpg'.format(img_root), recursive=True)
            if not self.paths:  # if no .jpg files found, try .JPEG
                self.paths = glob('{:s}/**/*.JPEG'.format(img_root), recursive=True)
        else:
            self.paths = glob('{:s}/**/*.jpg'.format(img_root), recursive=True)
            if not self.paths:  # if no .jpg files found, try .JPEG
                self.paths = glob('{:s}/**/*.JPEG'.format(img_root), recursive=True)

        # Try both .jpg and .JPEG for masks
        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))
        if not self.mask_paths:  # if no .jpg files found, try .JPEG
            self.mask_paths = glob('{:s}/*.JPEG'.format(mask_root))

        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        # Convert mask to grayscale (L) instead of RGB
        mask = self.mask_transform(mask.convert('L'))  # [1, H, W]
        # Expand mask to 3 channels to match input image
        mask = mask.expand(3, -1, -1)  # [3, H, W]
        # Apply mask to image and return
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)