import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    './autodl-tmp/data-imagenetv2/imagenetv2-matched-frequency-format-val',
)
OUTPUT_DIR = './dataset-quant'
# 2^8, not 8192: with weights at 2^13, images at 8192 overflow the int32 accumulators.
# Biases are at this scale too (see gen_weights.py)
QUANTIZATION_SCALE = 256

CLASSES = models.AlexNet_Weights.DEFAULT.meta['categories']

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def quantize(img):
    return (img.numpy().flatten() * QUANTIZATION_SCALE).astype(np.int32)


def generate_header(name, values, label, prediction, score, root=OUTPUT_DIR):
    os.makedirs(root, exist_ok=True)

    filename = f'{root}/{name}.h'
    header_guard = name.upper()
    with open(filename, 'w') as f:
        f.write(f'#ifndef __{header_guard}_H__\n')
        f.write(f'#define __{header_guard}_H__\n\n')
        f.write(f'// class: {label} ({CLASSES[label]})\n')
        f.write(f'// predicted class: {prediction} (val = {score}, {CLASSES[prediction]})\n')
        f.write(f'const int {name}[{values.size}] = {{\n')
        f.write(',\n'.join(map(str, values)))
        f.write('\n};\n')
        f.write('\n#endif')

    print(f'[+] image generated in {filename} file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantized ImageNetV2 images as .h files')
    parser.add_argument('--num-images', type=int, default=10, help='number of images to generate')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='directory to write .h files to')
    args = parser.parse_args()

    dataset = torchvision.datasets.ImageFolder(DATA_PATH, transform=transform)
    # the dataset is sorted by class: evenly spaced picks cover different classes
    picks = [n * len(dataset) // args.num_images for n in range(args.num_images)]
    samples = [dataset[i] for i in picks]
    # ImageFolder sorts the folders as strings ('0', '1', '10', ...): the folder
    # name is the ImageNet class, not ImageFolder's index
    labels = [int(dataset.classes[target]) for _, target in samples]
    imgs = torch.stack([img for img, _ in samples])

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()
    with torch.no_grad():
        scores, predictions = model(imgs).max(dim=1)
        scores = (scores * QUANTIZATION_SCALE).round().long().tolist()
        predictions = predictions.tolist()

    for n, (img, label, prediction, score) in enumerate(zip(imgs, labels, predictions, scores), start=1):
        generate_header(f'image{n}', quantize(img), label, prediction, score, root=args.output_dir)
