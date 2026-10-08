# 3layer_dense: MNIST image denoising

The model takes a noisy MNIST image and outputs a new image that should match the original. It is trained on pixels only: the target is the clean original image and the loss is mean squared pixel error. This is a denoising / reconstruction task, not classification or prediction. Digit labels are used only to split the data evenly and for the optional accuracy check described under Metrics.

## Setup

| item | value |
| --- | --- |
| data | MNIST, all 70,000 images, stratified subsets of [2500, 5000] images |
| split | 70% train / 10% validation / 20% test (stratified) |
| noise | Gaussian, sigma = 0.15, clipped to [0, 1]. Added to the input images of train, validation and test (different random noise per split). The targets are always the clean originals. |
| epochs | [50, 100] |
| optimizer | Adam, learning rate 0.01, batch size 64 |
| seed | 42 (same for every run) |
| runtime | CPU, TensorFlow 2.21.0, Python 3.13.16 |

## Model

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 128)            │       100,480 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 784)            │        50,960 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 479,090 (1.83 MB)
 Trainable params: 159,696 (623.81 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 319,394 (1.22 MB)

## Results

`noisy` columns are the baseline (the noisy test image compared with the clean original). `denoised` columns are the model output compared with the clean original. The model only helps where `denoised` beats `noisy`. MSE: lower is better. PSNR, SSIM, Acc: higher is better.

| size | epochs | MSE noisy | MSE denoised | PSNR noisy | PSNR denoised | SSIM noisy | SSIM denoised | Acc noisy | Acc denoised | train_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 50 | 0.0121 | 0.0213 | 19.1866 | 17.2692 | 0.6542 | 0.7761 | 0.866 | 0.832 | 14.7 |
| 2500 | 100 | 0.0121 | 0.0228 | 19.1866 | 16.8991 | 0.6542 | 0.7746 | 0.866 | 0.828 | 27.8 |
| 5000 | 50 | 0.0121 | 0.0184 | 19.1951 | 17.9065 | 0.6558 | 0.8078 | 0.879 | 0.872 | 23.7 |
| 5000 | 100 | 0.0121 | 0.0195 | 19.1951 | 17.5778 | 0.6558 | 0.8 | 0.879 | 0.867 | 45.0 |

Lowest denoised MSE: size 5000, 50 epochs, MSE 0.0184 against a noisy baseline of 0.0121 (does not beat the baseline on MSE).

## Figures

![loss curves](figures/loss.png)

- size 2500, 50 epochs: [digits](figures/digits_n2500_ep50.png), [confusion matrices](figures/confusion_n2500_ep50.png)
- size 2500, 100 epochs: [digits](figures/digits_n2500_ep100.png), [confusion matrices](figures/confusion_n2500_ep100.png)
- size 5000, 50 epochs: [digits](figures/digits_n5000_ep50.png), [confusion matrices](figures/confusion_n5000_ep50.png)
- size 5000, 100 epochs: [digits](figures/digits_n5000_ep100.png), [confusion matrices](figures/confusion_n5000_ep100.png)

## Files

| path | content |
| --- | --- |
| `*.ipynb` | the notebook with all code |
| `logs/epoch_log.csv` | per-epoch loss, validation loss, learning rate and time for every run |
| `logs/results_summary.csv` | final metrics and training time per run |
| `logs/config.json` | seed, split, noise, learning rate, runtime and library versions |
| `logs/runs/*.csv` | the epoch log split into one file per run |
| `tables/results.csv`, `tables/results.md` | the results table |
| `figures/` | loss curves, digit examples and confusion matrices |

## Metrics

- **MSE**: mean squared pixel error against the clean original.
- **PSNR**: peak signal-to-noise ratio in dB, averaged per image.
- **SSIM**: structural similarity, averaged per image.
- **Acc** and the confusion matrices (secondary check): a small dense classifier (784-128-10) trained on clean images is applied to the original, noisy and denoised test images. It shows whether a denoised digit is still read as the right digit. The classifier is not part of the denoiser.

## Reproduce

Open the notebook in Google Colab and run all cells in order. The first logging cell clears `logs/`, so run the whole notebook from the top.
