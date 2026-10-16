import argparse
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import gen_dataset
import gen_weights

INT32_LIMIT = 2**31
SHIFT = gen_weights.QUANTIZATION_SCALE

# Integer model: int32 accumulators, results shifted right by 13 (floor).
# It runs in float64: every value is an integer and every partial sum stays below
# the layer's bound (the largest sum(|w|*|x|)), so while the bound is below 2^31
# the result is exact and floor(acc / 2^13) is the arithmetic shift.


def load_int_params(model):
    # same quantization and int32 truncation as the headers written by gen_weights.py
    params = {}
    for name, p in model.state_dict().items():
        scale = gen_weights.BIAS_SCALE if name.endswith('bias') else gen_weights.QUANTIZATION_SCALE
        params[name] = torch.from_numpy(gen_weights.quantize(p, scale).astype(np.int32)).double()
    return params


def quantize_images(imgs):
    # same quantization as the headers written by gen_dataset.py
    return torch.from_numpy(gen_dataset.quantize(imgs)).double().view(imgs.shape)


def conv(x, p, layer, stride, pad):
    w, b = p[f'{layer}.weight'], p[f'{layer}.bias']
    acc = F.conv2d(x, w, stride=stride, padding=pad)
    bound = F.conv2d(x.abs(), w.abs(), stride=stride, padding=pad).max().item()
    return F.relu(torch.floor(acc / SHIFT) + b.view(1, -1, 1, 1)), bound


def fc(x, p, layer, relu=True):
    w, b = p[f'{layer}.weight'], p[f'{layer}.bias']
    acc = x @ w.T
    bound = (x.abs() @ w.abs().T).max().item()
    out = torch.floor(acc / SHIFT) + b
    return (F.relu(out) if relu else out), bound


def maxpool(x):
    return F.max_pool2d(x, kernel_size=3, stride=2)


def int_forward(x, p):
    bounds = {}
    x, bounds['conv1'] = conv(x, p, 'features.0', stride=4, pad=2)
    x = maxpool(x)
    x, bounds['conv2'] = conv(x, p, 'features.3', stride=1, pad=2)
    x = maxpool(x)
    x, bounds['conv3'] = conv(x, p, 'features.6', stride=1, pad=1)
    x, bounds['conv4'] = conv(x, p, 'features.8', stride=1, pad=1)
    x, bounds['conv5'] = conv(x, p, 'features.10', stride=1, pad=1)
    x = maxpool(x).flatten(1)
    x, bounds['fc6'] = fc(x, p, 'classifier.1')
    x, bounds['fc7'] = fc(x, p, 'classifier.4')
    x, bounds['fc8'] = fc(x, p, 'classifier.6', relu=False)
    return x, bounds


def top_k(top5, labels, k):
    # % of images whose label is among the first k predictions
    return 100 * (top5[:, :k] == labels.view(-1, 1)).any(dim=1).float().mean().item()


def spare_bits(bound):
    # negative when the accumulator can overflow int32
    return math.floor(math.log2(INT32_LIMIT / bound))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the integer AlexNet with the float one on ImageNetV2')
    parser.add_argument('--num-images', type=int, default=10000, help='number of images to evaluate')
    parser.add_argument('--batch-size', type=int, default=50, help='images per batch')
    parser.add_argument('--workers', type=int, default=8, help='image loading processes')
    args = parser.parse_args()

    dataset = torchvision.datasets.ImageFolder(gen_dataset.DATA_PATH, transform=gen_dataset.transform)
    # same evenly spaced picks as gen_dataset.py
    picks = [n * len(dataset) // args.num_images for n in range(args.num_images)]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, picks),
                                         batch_size=args.batch_size, num_workers=args.workers)
    # ImageFolder sorts the folders as strings: the folder name is the ImageNet class
    classes = torch.tensor([int(c) for c in dataset.classes])

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()
    params = load_int_params(model)

    float_top5, int_top5, labels, worst = [], [], [], {}
    with torch.no_grad():
        for imgs, targets in loader:
            out, bounds = int_forward(quantize_images(imgs), params)
            float_top5.append(model(imgs).topk(5).indices)
            int_top5.append(out.topk(5).indices)
            labels.append(classes[targets])
            for name, bound in bounds.items():
                worst[name] = max(worst.get(name, 0), bound)
            print(f'[+] {sum(map(len, labels))}/{len(picks)} images')

    float_top5, int_top5, labels = torch.cat(float_top5), torch.cat(int_top5), torch.cat(labels)
    agreement = 100 * (float_top5[:, 0] == int_top5[:, 0]).float().mean().item()

    print(f'[+] {len(labels)} images, {len(labels.unique())} classes')
    print(f'    top-1 agreement (int vs float): {agreement:6.2f} %')
    print(f'    float top-1 / top-5 accuracy:   {top_k(float_top5, labels, 1):6.2f} % / {top_k(float_top5, labels, 5):6.2f} %')
    print(f'    int   top-1 / top-5 accuracy:   {top_k(int_top5, labels, 1):6.2f} % / {top_k(int_top5, labels, 5):6.2f} %')

    print('[+] int32 headroom (worst image): bound, spare bits')
    for name, bound in worst.items():
        bits = spare_bits(bound)
        print(f'    {name:6s} {int(bound):12d}  {bits if bits >= 0 else "OVERFLOW"}')

    if any(spare_bits(bound) < 0 for bound in worst.values()):
        print('[-] an int32 accumulator can overflow')
        sys.exit(1)
