import argparse
import torchvision.models as models

from gen_weights import quantize, generate_header, QUANTIZATION_SCALE, BIAS_SCALE

# Splits the quantized conv weights for a multi-PE AlexNet where every conv layer is split
# along C_in. PE i of conv layer L gets the input channels [i*C_in/n, (i+1)*C_in/n) of every
# output channel, so its weights keep the OIHW order with a smaller I: [C_out][C_in/n][K][K].
# Biases are only needed by the join that adds up the PEs' partial sums, so they are whole.
# The conv1 input is split the same way by gen_dataset.py --split-channels.

OUTPUT_DIR = './slices'

# conv layer: features index in the state_dict
CONV_LAYERS = {1: 0, 2: 3, 3: 6, 4: 8, 5: 10}
DEFAULT_PES = '3,8,4,6,4'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split the quantized AlexNet conv weights per PE, along C_in')
    parser.add_argument('--pes', default=DEFAULT_PES, help='PEs of conv1..conv5, comma separated')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='directory to write .h files to')
    args = parser.parse_args()

    pes = [int(n) for n in args.pes.split(',')]
    if len(pes) != len(CONV_LAYERS):
        raise SystemExit(f'--pes needs {len(CONV_LAYERS)} values, got {len(pes)}')

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    model.eval()
    state = model.state_dict()

    for (layer, feat), n_pes in zip(CONV_LAYERS.items(), pes):
        w = quantize(state[f'features.{feat}.weight'], QUANTIZATION_SCALE)  # [C_out][C_in][K][K]
        b = quantize(state[f'features.{feat}.bias'], BIAS_SCALE)

        c_in = w.shape[1]
        if c_in % n_pes:
            raise SystemExit(f'conv{layer}: C_in = {c_in} is not divisible by {n_pes} PEs')

        d = c_in // n_pes
        for pe in range(n_pes):
            generate_header(f'c{layer}_{pe}_weight', w[:, pe*d:(pe + 1)*d], root=args.output_dir)

        generate_header(f'bias{layer}', b, root=args.output_dir)
