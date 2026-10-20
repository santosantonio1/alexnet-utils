# AlexNet

This repo builds on [torchvision's pretrained AlexNet](https://pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html)
(`AlexNet_Weights.IMAGENET1K_V1`), trained on ImageNet-1k (1000 classes). It exists to support
bare-metal C experiments with AlexNet: tooling here quantizes the model, and ImageNetV2 test
images, into fixed-point C header files, and checks the quantized model against the float one.

## Project layout

| File | Purpose |
|---|---|
| `download_dataset.py` | Downloads ImageNetV2 (MatchedFrequency: 10,000 images, 10 per class) into `autodl-tmp/data-imagenetv2/`. |
| `gen_weights.py` | Exports the model's quantized weights as `.h` files. |
| `gen_dataset.py` | Exports quantized ImageNetV2 images as `.h` files. |
| `validate.py` | Runs the integer (quantized) model against the float one on ImageNetV2 and checks for int32 overflow. |
| `gen_slices.py` | Exports the quantized conv weights split per PE, along C_in, for a multi-PE AlexNet. |

There is no fused model: AlexNet has no BatchNorm layers to fuse.

Generated artifacts (`weights/`, `dataset-quant/`, `slices/`, `autodl-tmp/`) are gitignored — they're all
reproducible from the scripts above. The pretrained weights are downloaded by torchvision into
`~/.cache/torch` on first use.

## Setup

```sh
pip install torch torchvision numpy
```

## Usage

**1. Get the dataset** (needed for `gen_dataset.py` and `validate.py`):

```sh
python download_dataset.py
```

**2. Export quantized model weights as C headers:**

```sh
python gen_weights.py [--output-dir ./weights] [--layout oihw|hwio]
```

Writes one `.h` file per parameter tensor, named after the `state_dict` key
(`features.0.weight` → `features_0_weight`), quantized to fixed-point `int32`:
weights ×8192, biases ×256.
`--layout` picks the weight order: `oihw` (default), PyTorch's native order, or `hwio`.

**3. Export quantized test images as C headers:**

```sh
python gen_dataset.py [--num-images 10] [--output-dir ./dataset-quant] [--split-channels N]
```

Writes `image1.h` .. `imageN.h`, each holding a normalized, fixed-point `int32` (×256)
ImageNetV2 image in PyTorch's CHW layout, preprocessed as AlexNet expects (resize 256,
center-crop 224, ImageNet mean/std). The images are evenly spaced over the dataset, so they
come from different classes. Each header comments the true class and the float model's
predicted class (index + name) for that image. `--split-channels N` also writes each image
as `N` contiguous blocks of channels, `imageK_c0.h` .. `imageK_c<N-1>.h`: one per conv1 PE
when conv1 is split along C_in (see `gen_slices.py`).

**4. Validate the quantized model:**

```sh
python validate.py [--num-images 10000] [--batch-size 50] [--workers 8]
```

Runs the integer model and the float model on the same images and reports the top-1
agreement between them, the top-1/top-5 accuracy of both, and the int32 headroom of each
layer on the worst image. It exits with 1 if any int32 accumulator can overflow.

**5. Split the weights per PE** (multi-PE AlexNet, each conv layer split along C_in):

```sh
python gen_slices.py [--pes 3,8,4,6,4] [--output-dir ./slices]
```

`--pes` is the number of PEs of conv1..conv5. PE `i` of conv layer `L` gets a contiguous block
of `C_in/n` input channels of every output channel, written to `cL_i_weight.h` as
`[C_out][C_in/n][K][K]` (OIHW with a smaller I). The conv biases are written whole to
`biasL.h`, for the join that adds up the PEs' partial sums. The quantization is
`gen_weights.py`'s, so the values are the same as in its `oihw` headers. The conv1 input is
split with `gen_dataset.py --split-channels`.

## Quantization scheme

- Weights are scaled by `8192` (`QUANTIZATION_SCALE` in `gen_weights.py`).
- Images and biases are scaled by `256` (`QUANTIZATION_SCALE` in `gen_dataset.py`,
  `BIAS_SCALE` in `gen_weights.py`).
- All values are cast to `int32`, matching a fixed-point inference implementation on the
  embedded target: a conv/fc output is `(sum(w*x) >> 13) + bias`, followed by ReLU
  (except on the last layer).

A single 8192 scale for everything overflows the int32 accumulators. With images at ×256,
the largest `sum(|w|*|x|)` over all 10,000 ImageNetV2 images stays below 2^31 in every layer
(conv2 and conv3 are the closest, at about 60%). On ImageNetV2, the integer model agrees with
the float one on 99.54% of the top-1 predictions (top-1 accuracy 43.48% vs 43.49%).
