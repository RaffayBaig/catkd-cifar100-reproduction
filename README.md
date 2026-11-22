# CAT-KD Reproduction — CT-468 Deep Learning

Reproduction of **"Class Attention Transfer Based Knowledge Distillation"**
(Guo, Yan, Li & Lin, CVPR 2023) on CIFAR-100, teacher **ResNet32×4** → student **ResNet8×4**.

| Name | Roll No. |
|---|---|
| Umair Hasan | AI-23018 |
| Amna Amjad | AI-23031 |
| Mirza Abdul Raffay Baig | AI-23033 |

Submitted to Miss Murk Marvi, Course Instructor, Deep Learning (CT-468).

## Results

**Reproduction** (CIFAR-100 top-1 accuracy, %, 240 epochs, one run):

| Model | Paper | Ours | Gap |
|---|---|---|---|
| Teacher ResNet32×4 | 79.42 | 79.42 | 0.00 |
| CAT-KD student ResNet8×4 (best epoch) | 76.91 | **77.02** | +0.11 |
| CAT-KD student ResNet8×4 (last epoch) | — | 76.71 | — |

**Own experiment: CAT-KD with only 20% of the training data** (100 images per class, same images for both runs):

| Training data | Student alone | CAT-KD | Gain from CAT-KD |
|---|---|---|---|
| 100% | 72.50 (paper) | 77.02 | +4.52 |
| 20% | 54.15 | 66.18 | **+12.03** |

With less data, the teacher's class attention helps almost three times as much.

## Repository contents

```
catkd-reproduction/
├── README.md                     this file
├── requirements.txt              Python packages
├── code/
│   ├── CATKD_reproduction.ipynb  Colab notebook with all runs and their outputs
│   └── catkd.py                  the full method: models, CAMs, CAT loss, training
└── results/
    ├── accuracy_curves.png       test accuracy per epoch for all three runs
    ├── predictions.png           student predictions on 8 test images
    └── logs/                     per-epoch training logs (CSV)
```

**Logs** (`epoch, lr, loss, ce, aux, test_acc, best, sec`):
- `catkd_ep240_b15_r2_sub0.2_log.csv`: CAT-KD, 20% data, epochs 0–239
- `ce_ep240_b15_r2_sub0.2_log.csv`: student alone, 20% data, epochs 0–239
- `catkd_ep240_b15_r2_log.csv`: main CAT-KD run, epochs 0–239 (resumed at epoch 84 after the free GPU limit;
  the log continues in the same file)

## Requirements

Python 3, PyTorch ≥ 2.3 and torchvision (pre-installed on Google Colab). A GPU is strongly recommended: about
43 s per epoch on a free Colab T4, so about 2.9 h for 240 epochs on full data and about 45 min on 20% of the data.

## How to re-run (Google Colab)

Open `code/CATKD_reproduction.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the
cells from the top. Step by step:

1. Mount Google Drive, so logs and checkpoints survive disconnects:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Download the pretrained teacher (the same checkpoint the authors used, from MDistiller):
   ```bash
   !wget -q https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar
   !mkdir -p download_ckpts && tar xf cifar_teachers.tar -C download_ckpts
   ```
3. Put `catkd.py` in `/content` (upload it, or run the notebook cell that starts with `%%writefile catkd.py`).
4. Train. CIFAR-100 downloads automatically the first time.
   ```bash
   # Main reproduction: CAT-KD, beta = 15, 2x2 CAMs, no normalisation (authors' config)
   !python catkd.py --method catkd --epochs 240 --out /content/drive/MyDrive/catkd_runs

   # Own experiment: 20% of the training images per class
   !python catkd.py --method catkd --subset 0.2 --out /content/drive/MyDrive/catkd_runs
   !python catkd.py --method ce    --subset 0.2 --out /content/drive/MyDrive/catkd_runs
   ```

The script first prints the teacher's accuracy (about 79.42), then one line per epoch, and ends with
`FINAL [<run name>] best top-1 = XX.XX`.

**Resuming:** each run saves `<run name>_last.pth` after every epoch. After a disconnect, redo steps 1–3 and run
the **same command**: it prints `Resumed from epoch N` and continues. Changing any setting starts a new run.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--method` | `catkd` | `catkd` (CE + β·CAT), `kd` (Hinton KD, T = 4) or `ce` (student alone) |
| `--epochs` | 240 | Training length; the learning-rate drops (÷10 at 150/180/210 of 240) scale with it |
| `--beta` | 15 | CAT loss weight β (from the authors' `res32x4_res8x4.yaml`) |
| `--cam_res` | 2 | Size the CAMs are average-pooled to before matching |
| `--normalize` | off | L2-normalise each pooled CAM (authors use it only when architectures differ) |
| `--subset` | 1.0 | Fraction of training images kept per class, fixed seed (own experiment) |
| `--binarize` | off | Binarise the teacher's CAMs (idea from the paper's Table 3) |
| `--amp` | off | FP16 mixed precision: faster but can overflow, so off by default |
| `--bs`, `--lr` | 64, 0.05 | Batch size and initial learning rate (paper settings) |
| `--out` | `runs` | Folder for logs and checkpoints |

## How the code maps to the paper

- **`ResNet.forward`**: the FC layer is applied as a 1×1 convolution *before* global average pooling, giving one
  class activation map (CAM) per class (B × 100 × 8 × 8). Averaging each map gives exactly the original logits.
- **`cat_loss`**: the CAT loss. CAMs of all classes are average-pooled to 2×2, optionally normalised, and compared
  with mean squared error, as in the authors' `CAT_loss`.
- **Training loop**: L = L_CE + β·L_CAT, SGD (momentum 0.9, weight decay 5e-4), batch 64, learning rate 0.05
  divided by 10 at epochs 150, 180 and 210.

## Notes

- **PyTorch ≥ 2.6:** the teacher checkpoint is loaded with `weights_only=False`, because it also stores NumPy
  values. It comes from the official MDistiller release.
- **β matters:** with β = 400 (a value from the paper's appendix, meant for normalised CAMs) training diverged to
  NaN within 4 iterations. The authors' config for this pair uses β = 15, which is stable. The script stops at once
  if the loss becomes NaN.
- **Single runs:** each result comes from one seed; CIFAR-100 runs usually vary by a few tenths of a point.

## References

- Z. Guo, H. Yan, H. Li, X. Lin. *Class Attention Transfer Based Knowledge Distillation.* CVPR 2023,
  pp. 11868–11877. https://arxiv.org/abs/2304.12777
- Official CAT-KD code: https://github.com/GzyAftermath/CAT-KD
- MDistiller (teacher checkpoints): https://github.com/megvii-research/mdistiller
