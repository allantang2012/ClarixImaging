import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

import opt
from net import PConvUNet
from util.io import load_ckpt

parser = argparse.ArgumentParser()
parser.add_argument('--snapshot', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--image', type=str, required=True, help='Path to the test image')
parser.add_argument('--mask', type=str, default='./masks/00000.jpg', help='Path to the mask image')
parser.add_argument('--image_size', type=int, default=64, help='Image size for testing')
parser.add_argument('--output', type=str, default='result.png', help='Output image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and transform image
size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

# Load image
image = Image.open(args.image)
image = img_transform(image.convert('RGB')).unsqueeze(0).to(device)

# Load mask
mask = Image.open(args.mask)
mask = mask_transform(mask.convert('L'))
mask = mask.expand(3, -1, -1)  # Expand to 3 channels
mask = mask.unsqueeze(0).to(device)

# Load model
model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])
model.eval()

# Generate inpainted image
with torch.no_grad():
    output, _ = model(image * mask, mask)
    output_comp = mask * image + (1 - mask) * output

# Save results
def to_image(x):
    x = F.interpolate(x, size=size)
    x = x[0].cpu().permute(1, 2, 0).numpy()
    x = ((x * opt.STD + opt.MEAN) * 255).astype(np.uint8)
    return Image.fromarray(x)

# Create comparison image
masked_image = to_image(image * mask)
inpainted = to_image(output_comp)
original = to_image(image)

# Create a side-by-side comparison
result = Image.new('RGB', (size[0] * 3, size[1]))
result.paste(original, (0, 0))
result.paste(masked_image, (size[0], 0))
result.paste(inpainted, (size[0] * 2, 0))
result.save(args.output)

print(f"Results saved to {args.output}")
print("Left: Original | Middle: Masked | Right: Inpainted")