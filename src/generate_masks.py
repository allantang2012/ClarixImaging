import os

import numpy as np

from PIL import Image

import random



def create_rectangular_mask(width, height, min_size=20, max_size=50):

    """Create a mask with a random rectangular hole"""

    mask = np.ones((height, width), np.uint8) * 255  # White background



    # Random rectangle parameters

    rect_width = random.randint(min_size, max_size)

    rect_height = random.randint(min_size, max_size)



    # Random position

    x = random.randint(0, width - rect_width)

    y = random.randint(0, height - rect_height)



    # Create rectangular hole (black)

    mask[y:y+rect_height, x:x+rect_width] = 0



    return mask



def generate_masks(num_masks, output_dir, width=64, height=64):

    """Generate multiple masks and save them"""

    os.makedirs(output_dir, exist_ok=True)



    for i in range(num_masks):

        mask = create_rectangular_mask(width, height)

        mask_img = Image.fromarray(mask)

        mask_img.save(os.path.join(output_dir, f'mask_{i:05d}.png'))

        if i % 1000 == 0:

            print(f'Generated {i} masks')



# Create masks directory

masks_dir = 'masks'

if not os.path.exists(masks_dir):

    os.makedirs(masks_dir)



# Generate masks (Tiny ImageNet images are 64x64)

# Let's generate enough masks for training


generate_masks(num_masks=10000, output_dir=masks_dir, width=64, height=64)

