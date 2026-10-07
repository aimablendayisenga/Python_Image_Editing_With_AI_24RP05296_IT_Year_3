# Smart AI Photo Editor (Picsart-style)

**Module:** ITLPA701 Python and Fundamentals of AI | **Approach:** Deep Learning (CNN)
**Domains:** Image recognition + task automation (media editing)

## 1. Problem
Photos are often blurry, noisy, too dark or too small. Users do not know which fix to apply. This app **recognises the problem with a CNN** and **applies the right edit automatically**, like the AI tools in Picsart.

## 2. Setup
```
python -m venv venv
venv\Scripts\activate          (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Python 3.12 or 3.13 recommended. Works in VS Code, PyCharm, Jupyter or Spyder.

## 3. Dataset
The project includes the Berkeley Segmentation Dataset and Benchmark (BSDS500) archive at `data/archive/BSR_bsds500.tgz` and its 200 official training photographs at `data/train/`. The filenames in `data/train/` match all 200 files in the archive's `BSR/BSDS500/data/images/train/` split. This supplies clean source photos; the damage labels are generated synthetically by the project rather than downloaded. Dataset reference: Arbelaez, Maire, Fowlkes, and Malik, “Contour Detection and Hierarchical Image Segmentation,” IEEE TPAMI, 2011; the source paper is also included in the archive.

**Preprocessing:** images are converted to RGB and resized to a maximum side of 512 px. Source images are split 80/20 before generating 96x96 crops, so validation images do not occur in training. Pixels are scaled to [0,1]. Balanced synthetic classes are generated from clean photos: sharp (unchanged), blurry (Gaussian blur), noisy (Gaussian noise), and dark (brightness x0.15-0.5). Training uses 1,600 generated crops; classifier validation uses 400 crops.

## 4. Architecture
```
Photo -> 96x96 crops -> CNN (3 conv blocks) --+
                  \-> engineered features ----+-> dense layers -> sharp | blurry | noisy | dark
                       (sharpness, brightness, noise level, edge strength)
Recognised class -> auto fix: blurry -> unsharp mask | noisy -> TV denoise | dark -> gamma brighten
                  -> optional HD (2x) / Ultra HD (4x) with the super-resolution CNN (enhancer.py)
```
- When an uploaded image is predicted as noisy, the app automatically creates a denoised After preview. The Before image remains the untouched original.
- The editor also includes one-click style presets: natural, vivid, warm, cool, black & white, portrait, cinematic, soft matte, golden hour, editorial clean, teal & orange, film fade, moody, pastel, and high-key studio.
- **Data features engineered:** Laplacian variance, mean brightness, noise estimate, gradient strength.
- **Model features engineered:** Conv-BatchNorm-ReLU-MaxPool blocks (16, 32, 64 filters) + dropout 0.3.
- **Why CNN:** the task is visual recognition of spatial blur, noise, and illumination patterns; convolution layers learn local image features, while the four engineered measurements provide complementary cues. A separate residual CNN provides optional 2x super-resolution.

## 5. Parameters
| Parameter | Value | Reason |
|---|---|---|
| Learning rate | 5e-4 | Adam; selected by validation tuning against 1e-3 |
| Epochs / batch | 12 / 32 | best epoch kept by validation F1 |
| Filters | 16-32-64 | small, fast on CPU |
| Dropout | 0.3 | reduces overfitting |
| Engineered features | enabled | validation tuning compared enabled vs disabled |

The super-resolution CNN uses 8 residual blocks, 64 channels, a 5e-4 learning rate, 30 epochs, batch size 16, and 64x64 patches. Its block count and learning rate were selected from the recorded tuning trials.

## 6. Evaluation
- **Classifier:** accuracy, precision, recall, macro F1, and confusion matrix are computed on 400 synthetically damaged crops from held-out source images (`models/clf_metrics.json`).
- **Automation:** `python editor.py evaluate --data data/train` evaluates damage detection, whether the fixed image is then classified as sharp, and PSNR change (`models/auto_metrics.json`).
- **Upscaler:** PSNR/SSIM/RMSE are compared with bicubic on held-out images (`models/metrics.json`).
- **Tuning:** `python classifier.py tune --data data/train` compares learning rate and engineered-feature settings using validation F1, saves the trials to `models/clf_tuning.json`, and saves the selected classifier settings/model.
- **Interpretation:** read the recorded validation results below before presenting. The classifier scores measure performance on synthetic damage, not on independently collected real-world damaged photos.

### Recorded results and interpretation
All classifier and automation results below use held-out BSDS500 source images, but the image damage is synthetic.

| Evaluation | Result |
|---|---|
| Classifier validation (400 balanced crops) | accuracy 0.9325; macro precision 0.9344; macro recall 0.9325; macro F1 0.9321 |
| Upscaler validation (40 images) | CNN: PSNR 29.1243, SSIM 0.8746, RMSE 0.0375; bicubic: PSNR 27.1358, SSIM 0.8157, RMSE 0.0468 |
| Blurry correction (30 samples) | detection 0.967; task success 0.233; mean PSNR gain 0.06 dB |
| Noisy correction (30 samples) | detection 1.000; task success 0.567; mean PSNR gain 5.56 dB |
| Dark correction (30 samples) | detection 0.967; task success 0.833; mean PSNR gain 5.24 dB |

The classifier has strong overall results on the synthetic validation set; the confusion matrix shows most errors are sharp images classified as blurry or dark. Noise detection was perfect on this split. Automation is less successful than detection alone suggests: sharpening only brought 23.3% of blurry samples to the classifier's sharp class, so deblurring remains a known weakness. Denoising and brightening produced larger average PSNR gains. These results do not establish performance on real camera blur, noise, or lighting.

The classifier tuning trials (6 epochs per candidate) scored macro F1 0.8929 for (1e-3, features on), 0.9066 for (1e-3, features off), and 0.9125 for (5e-4, features on). The selected model was then trained for 12 epochs and reached validation macro F1 0.9321. See `models/clf_tuning.json` and `models/clf_config.json`.

For super-resolution, the CNN improves PSNR by 1.9885 dB and SSIM by 0.0589 over bicubic on the 40-image validation split; its RMSE is 0.0093 lower. The final checkpoint configuration matches the best recorded 10-epoch tuning candidate (8 blocks, 5e-4 learning rate), followed by 30-epoch training. See `models/tuning.json`, `models/config.json`, and `models/metrics.json`.

### Repeatable checks
Run the core pipeline tests with:
```
python -m unittest discover -s tests -v
```
These check engineered features, synthetic corruption, classifier output dimensions, editor transforms, and super-resolution output dimensions.

## 7. Run (in this order)
```
python classifier.py train --data data/train      (or: tune)
python editor.py evaluate --data data/train       # automation test
python enhancer.py train --data data/train        # optional, enables HD / Ultra HD
streamlit run app.py
```
For a complete reproducible assessment, run classifier tuning and automation evaluation as well as training. Saved model weights, configurations, and metrics are under `models/`.
The before/after demo uses `assets/AIMABLE BEFORE.png` and `assets/AIMABLE AFTER.jpg`; the app uses a softened, darkened version of `assets/AIMABLE AFTER.jpg` as its full-page background. The Streamlit top toolbar keeps its default appearance.

## 8. Responsible use
- **Risk 1:** misclassification leads to a wrong edit. *Reduction:* show confidence, let the user override the fix, compare with the original.
- **Risk 2:** upscaling invents details (faces, text). *Reduction:* warning in the app, not for evidence.
- **Risk 3:** limited dataset coverage and photo privacy. *Reduction:* evaluate on broader real-world photos before relying on predictions; uploaded photos are processed in the app and are not sent to GitHub.
- **Limit:** trained on synthetic damage, so real-world photos can behave differently.

## Rubric mapping
Environment/data/preprocessing: 2-3 | Features and model: 4 | Parameters: 5 | Implementation: `classifier.py`, `editor.py`, `enhancer.py` | Evaluation and tuning: 6 | Saved model: 7 | Demo: `app.py`
