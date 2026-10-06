# Lung Histopathology Classification with DenseNet-169

This project classifies lung histopathology images into three classes using a pretrained PyTorch DenseNet-169. The notebook is configured to use the Intel Arc A750 through PyTorch's XPU backend and includes training progress, evaluation metrics, and visual reports.

## Project Files

- `DenseNet169.ipynb`: end-to-end data loading, splitting, training, evaluation, charts, and optional online-sample prediction.
- `data/`: the main local dataset used for training and evaluation.
- `best_model.pt`: the best model checkpoint saved during the most recent training run.
- `holdout_300_manifest.csv`: filenames and labels for the balanced 300-image holdout.
- `external_lc25000_sample/`: downloaded prediction-only images, stored by class.
- `external_lc25000_manifest.csv`: source repository revision and filenames for the downloaded images.

## Classes and Data

The main dataset is expected in this layout:

```text
data/
  adenocarcinoma/
  squamous_cell_carcinoma/
  benign/
```

The current local dataset contains 15,000 images, 5,000 per class. Supported formats include JPEG, PNG, BMP, and TIFF. Images are converted to RGB and resized to 224 x 224 pixels. ImageNet mean and standard deviation are used for normalization.

Class indices are:

| Index | Class |
| --- | --- |
| 0 | adenocarcinoma |
| 1 | squamous_cell_carcinoma |
| 2 | benign |

## Environment

Verified environment:

- Windows, Python 3.14.8
- PyTorch 2.14.1 with XPU support
- TorchVision 0.29.1 with XPU support
- Intel Arc A750 Graphics selected as `xpu`
- NumPy 2.5.3, Pillow 12.3.0, scikit-learn 1.9.1
- Matplotlib 3.11.2, torchinfo 1.8.0, tqdm 4.70.1, ipykernel 7.4.0
- Intel oneAPI/SYCL runtime packages installed with the XPU build

The notebook checks XPU first, then CUDA, then CPU. It prints the selected device at startup and warns if it falls back to CPU. The Intel Radeon integrated graphics device is not selected by this notebook; the tested accelerator is the Arc A750.

Use the Python interpreter/kernel that has the `+xpu` PyTorch packages installed. After changing or installing PyTorch packages, restart the notebook kernel before running cells. A standard CPU-only PyTorch installation will not use the Arc GPU.

The notebook imports these direct Python packages: `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn`, `matplotlib`, `torchinfo`, and `tqdm`. The XPU PyTorch and TorchVision builds also require their matching Intel runtime dependencies.

## Model and Training

The model architecture and training settings are:

- DenseNet-169 with ImageNet-1K V1 pretrained weights.
- Classifier replaced with dropout (0.4) and a 3-output linear layer.
- 12,489,475 trainable parameters.
- AdamW optimizer; learning rate `2e-4`; weight decay `0.05`.
- Cross-entropy loss with label smoothing `0.1`.
- Batch size 32; maximum 15 epochs; one warmup epoch; early-stopping patience 4.
- Cosine learning-rate schedule and gradient clipping at 1.0.
- XPU automatic mixed precision and channels-last model/input memory format.
- Random horizontal/vertical flips, rotations, and color/contrast perturbations for training batches.

The dataset is loaded into memory before training. The decoded 15,000-image uint8 tensor uses about 2.1 GiB of RAM, in addition to model, Python, and GPU memory. Image decoding and CPU-side notebook work still use the CPU even while model operations run on the GPU.

## Split and Holdout

The notebook deterministically reserves 100 images from each class before creating any other split. These 300 images are excluded from training, validation, and the internal test split. Their relative paths and labels are written to `holdout_300_manifest.csv`.

The remaining 14,700 images are split with stratification:

| Split | Images |
| --- | ---: |
| Train | 11,760 |
| Validation | 1,470 |
| Internal test | 1,470 |
| Final 300-image holdout | 300 |

The holdout is evaluated only after the best validation checkpoint is loaded. It is not used for training, augmentation, early stopping, or checkpoint selection.

## Run the Notebook

1. Open `DenseNet169.ipynb` in VS Code.
2. Select the Python 3.14.8 kernel that reports PyTorch `2.14.1+xpu`.
3. Restart the kernel if packages or the Python interpreter have changed.
4. Run all cells from top to bottom.

The training cell displays batch progress, running loss/accuracy, and GPU memory. After training, the notebook shows training/validation curves, epoch time and accelerator memory, internal-test reports and confusion matrices, final holdout results, and labeled example predictions.

The final online-sample cell requires internet access the first time it runs. It downloads 200 images for each class from the dataset listed below, saves them under `external_lc25000_sample/`, writes a manifest, checks exact resized-image matches against the local in-memory data, and performs inference only with `best_model.pt`. It does not train or update the model with those downloaded files.

## Recorded Results

The latest recorded run completed all 15 epochs on the Arc A750, using about 2.59 GB peak GPU memory during training.

| Evaluation | Correct | Accuracy |
| --- | ---: | ---: |
| Internal test, 1,470 images | 1,470 / 1,470 | 100% |
| Reserved holdout, 300 images | 300 / 300 | 100% |
| Downloaded prediction sample, 600 images | 600 / 600 | 100% |

The internal test had 490 images per class. The final holdout had 100 per class. The online sample had 200 per class. The notebook also prints precision, recall, F1, loss, and confusion matrices for these evaluations.

## Important Evaluation Caveat

The downloaded sample comes from the Hugging Face `marlygotti/LC25000` repository, which identifies itself as an LC25000 mirror. Its 15,000-row three-class layout matches the size and class structure of the local data. A check found no exact resized-pixel matches among the selected 600 images, but that does **not** prove they are independent of the local set: images can be re-encoded, transformed, or derived from the same underlying source collection.

Therefore, the 600-image result is a prediction smoke test on a source-matched mirror, not an independent estimate of generalization. The local holdout is also an image-level random sample; the dataset does not provide patient/slide identifiers here. For a stronger research claim, evaluate on a separately collected cohort and split by patient or slide where those identifiers are available. These results are for research/educational use and are not a medical diagnostic tool.

Dataset mirror: [marlygotti/LC25000 on Hugging Face](https://huggingface.co/datasets/marlygotti/LC25000). Check its current dataset card and license/attribution terms when redistributing images.
