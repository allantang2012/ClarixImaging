import argparse

import torch

from torchvision import transforms

from torch.utils.data import Dataset

import os

from PIL import Image

import random

from net import PConvUNet

from util.io import load_ckpt

import opt

from torchvision.utils import save_image



parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, required=True, help='path to Tiny ImageNet')

parser.add_argument('--mask', type=str, required=True, help='path to masks')

parser.add_argument('--snapshot', type=str, default='', help='path to model weights')

parser.add_argument('--image_size', type=int, default=64, help='image size')

parser.add_argument('--result', type=str, default='result.jpg', help='output file name')

args = parser.parse_args()



class TinyImageNetDataset(Dataset):

    def __init__(self, root, mask_root, img_transform, mask_transform):

        self.root = root

        self.mask_root = mask_root

        self.img_transform = img_transform

        self.mask_transform = mask_transform



        # Get image paths

        self.image_paths = []

        for class_dir in os.listdir(root):

            class_path = os.path.join(root, class_dir, 'images')

            if os.path.isdir(class_path):

                for img_name in os.listdir(class_path):

                    if img_name.endswith('.JPEG'):

                        self.image_paths.append(os.path.join(class_path, img_name))



        # Get mask paths

        self.mask_paths = [os.path.join(mask_root, x) for x in os.listdir(mask_root)

                          if x.endswith('.png')]

        print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, idx):

        # Load image

        img_path = self.image_paths[idx]

        img = Image.open(img_path).convert('RGB')

        img = self.img_transform(img)



        # Load random mask and convert to 3 channels

        mask_path = random.choice(self.mask_paths)

        mask = Image.open(mask_path)

        mask = self.mask_transform(mask)

        # Repeat the mask in 3 channels

        mask = mask.repeat(3, 1, 1)



        return img, mask, os.path.basename(img_path)



def evaluate(model, dataset, device, output_path):

    print('Start evaluation...')

    model.eval()



    # Create output directory

    output_dir = os.path.splitext(output_path)[0]

    os.makedirs(output_dir, exist_ok=True)



    with torch.no_grad():

        for idx in range(min(100, len(dataset))):  # Process up to 100 images

            img, mask, img_name = dataset[idx]



            # Add batch dimension

            img = img.unsqueeze(0).to(device)

            mask = mask.unsqueeze(0).to(device)



            # Create input (masked image)

            masked = img * mask



            # Inpaint

            output = model(masked, mask)



            # Save results

            result = torch.cat([masked, output, img], dim=0)

            save_image(

                result,

                os.path.join(output_dir, f'{idx}_{img_name}.png'),

                nrow=3,

                normalize=True,

                range=(-1, 1)

            )



            if idx % 10 == 0:

                print(f'Processed {idx+1} images')



    print('Evaluation completed')



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')



    # Set up transforms

    size = (args.image_size, args.image_size)

    img_transform = transforms.Compose([

        transforms.Resize(size=size),

        transforms.ToTensor(),

        transforms.Normalize(mean=opt.MEAN, std=opt.STD)

    ])

    mask_transform = transforms.Compose([

        transforms.Resize(size=size),

        transforms.ToTensor()

    ])



    # Create dataset

    dataset_val = TinyImageNetDataset(

        args.root,

        args.mask,

        img_transform,

        mask_transform

    )



    # Load model

    model = PConvUNet().to(device)

    if args.snapshot:

        print(f'Loading model from: {args.snapshot}')

        load_ckpt(args.snapshot, [('model', model)])

    else:

        print('No model snapshot provided, using untrained model')



    # Evaluate

    evaluate(model, dataset_val, device, args.result)



if __name__ == '__main__':

    main()

