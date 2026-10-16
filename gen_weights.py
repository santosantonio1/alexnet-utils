import argparse
import os
import numpy as np
import torchvision.models as models

QUANTIZATION_SCALE = 8192
# biases are added after the >> 13, so they use the image scale (see gen_dataset.py)
BIAS_SCALE = 256
OUTPUT_DIR = './weights'


def quantize(params, scale=QUANTIZATION_SCALE):
    return params.detach().numpy() * scale


def to_hwio(params):
    if params.ndim == 4:
        return params.transpose(2, 3, 1, 0)
    if params.ndim == 2:
        return params.T
    return params


def generate_header(name, params, root=OUTPUT_DIR, layout='oihw'):
    os.makedirs(root, exist_ok=True)

    if layout == 'hwio':
        params = to_hwio(params)
        
    name = name.replace('.', '_')
    header_guard = name.upper()
    values = params.flatten().astype(np.int32)

    filename = f'{root}/{name}.h'
    with open(filename, 'w') as f:
        f.write(f'#ifndef __{header_guard}_H__\n')
        f.write(f'#define __{header_guard}_H__\n\n')
        f.write(f'// {name}\n')
        f.write(f'// {params.shape}\n')
        f.write(f'const int {name}[] = {{\n')
        f.write(',\n'.join(map(str, values)))
        f.write('\n};\n')
        f.write('\n#endif')

    print(f'[+] parameters generated in {filename} file!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantized AlexNet weights as .h files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='directory to write .h files to')
    parser.add_argument('--layout', choices=['hwio', 'oihw'], default='oihw',
                        help='weight order: hwio (TensorFlow) or oihw (PyTorch)')
    args = parser.parse_args()

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()

    for name, params in model.state_dict().items():
        scale = BIAS_SCALE if name.endswith('bias') else QUANTIZATION_SCALE
        generate_header(name, quantize(params, scale), root=args.output_dir, layout=args.layout)
