# CycleGAN: MRI → CT Translation

An unpaired image-to-image translation model that converts brain **MRI** slices into synthetic **CT**-like images (and back), built from scratch in TensorFlow/Keras. CycleGAN needs no pixel-aligned MRI/CT pairs, which are rare in clinical data: it learns the mapping from two unpaired image collections.

## How it works

Two generators and two discriminators are trained together:

| Network | Role |
|---|---|
| `G_MtoC` | Generator: MRI → CT |
| `G_CtoM` | Generator: CT → MRI |
| `D_C` | Discriminator: real vs. generated CT |
| `D_M` | Discriminator: real vs. generated MRI |

**Losses**

- **Adversarial loss** (binary cross-entropy): pushes each generator to produce images its discriminator can't tell apart from real ones.
- **Cycle-consistency loss** (L1): an image translated to the other domain and back should match the original,
  `‖G_CtoM(G_MtoC(mri)) − mri‖₁ + ‖G_MtoC(G_CtoM(ct)) − ct‖₁`.
- **Total generator loss** = adversarial + **10 ×** cycle-consistency.

**Architecture**

- *Generator:* a convolutional encoder–decoder: two strided `Conv2D` downsampling layers (64 → 128 filters) and two `Conv2DTranspose` upsampling layers, ending in `tanh`.
- *Discriminator:* two strided `Conv2D` layers with LeakyReLU (0.2), then a dense sigmoid output.

**Data pipeline:** grayscale PNG slices are resized to 256 × 256 and scaled to `[-1, 1]` with `tf.data`.

## Repository structure

```
cycleGAN.ipynb   # data loading, models, losses, training loop and visualizations
```

## Running it

1. Get an unpaired MRI/CT slice dataset (the notebook installs the Kaggle CLI for downloading) and place the images in two folders, one for MRI and one for CT.
2. Point `load_dataset(train_mri_dir, train_ct_dir)` at those folders.
3. Run the notebook. It trains for 15 epochs and plots real, translated and reconstructed images as it goes.

**Requirements:** Python 3, TensorFlow 2.x, NumPy, Matplotlib.

## Tech stack

`TensorFlow` · `Keras` · `CycleGAN` · `Cycle-Consistency Loss` · `Medical Imaging`
