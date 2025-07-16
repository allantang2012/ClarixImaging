import os

from PIL import Image



def check_data_structure():

    # Check training images

    train_path = r"C:\Tiny ImageNet\tiny-imagenet-200\tiny-imagenet-200\train"

    print("Checking training data...")



    # Count images

    image_count = 0

    for root, dirs, files in os.walk(train_path):

        for file in files:

            if file.endswith(('.JPEG', '.jpeg', '.jpg', '.png')):

                image_count += 1

                if image_count == 1:  # Check first image

                    img_path = os.path.join(root, file)

                    img = Image.open(img_path)

                    print(f"First image size: {img.size}")

    print(f"Total training images found: {image_count}")



    # Check masks

    masks_path = "./masks"

    print("\nChecking masks...")

    mask_count = len([f for f in os.listdir(masks_path) if f.endswith('.png')])

    if mask_count > 0:

        first_mask = Image.open(os.path.join(masks_path, os.listdir(masks_path)[0]))

        print(f"First mask size: {first_mask.size}")

    print(f"Total masks found: {mask_count}")



if __name__ == '__main__':

    check_data_structure()