# Assignment 3: MNIST GAN

The GAN is integrated into `app/main.py`, alongside the original text,
embedding, and CNN endpoints. Source is in `app/gan/`; the inference
checkpoint is `models/mnist_generator.pt`.

## Architecture

| Generator operation | Output shape |
| --- | --- |
| Noise input | `(B, 100)` |
| Linear and reshape | `(B, 128, 7, 7)` |
| ConvTranspose2d(128, 64, 4, 2, 1), BatchNorm2d, ReLU | `(B, 64, 14, 14)` |
| ConvTranspose2d(64, 1, 4, 2, 1), Tanh | `(B, 1, 28, 28)` |

| Discriminator operation | Output shape |
| --- | --- |
| Image input | `(B, 1, 28, 28)` |
| Conv2d(1, 64, 4, 2, 1), LeakyReLU(0.2) | `(B, 64, 14, 14)` |
| Conv2d(64, 128, 4, 2, 1), BatchNorm2d, LeakyReLU(0.2) | `(B, 128, 7, 7)` |
| Flatten, Linear, Sigmoid | `(B, 1)` |

The discriminator returns probabilities and training uses BCELoss. Real
images are normalized to `[-1, 1]` to match Tanh. Training alternates
discriminator and non-saturating generator updates.

## Included training run

The actual original run used all 60,000 MNIST training images, 10 epochs,
batch size 128, seed 42, Adam learning rate 0.0002, and betas (0.5, 0.999).
The generator has 765,761 parameters; the discriminator has 138,817.
Logs and fixed-noise samples are retained in `results/assignment3/`.
The original config records the standalone run's paths.

Current training command:

```bash
uv run python -m training.train_gan --epochs 20 --seed 42 --device cpu
```

The default checkpoint is `models/mnist_generator.pt`, with other outputs in
`results/assignment3/`. Use `--checkpoint-path` and `--output-dir` to keep a
new run separate. Restart the API after changing weights.

## Generate images

Send `{"count":4,"seed":42}` to POST `/gan/generate`.
GET `/gan/generate.png?count=16&seed=42` returns a viewable grid.
A missing checkpoint returns HTTP 503. `GAN_MODEL_PATH` overrides the
checkpoint location independently of the CNN's model path.

This model generates digits from noise and cannot select a requested class.
Some outputs remain ambiguous; integration does not claim a new FID,
digit-classifier accuracy, or quality benchmark.

## Theory and verification

The seven worked answers are in `theory_answers.md` and `theory_answers.pdf`.
The PDF and weights are preserved from the earlier English package.
Historical standalone checks are in
`results/assignment3/STANDALONE_VALIDATION.md`; current combined checks are in
`results/integration/VERIFICATION.md`.
