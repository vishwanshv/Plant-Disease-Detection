# 🌿 Plant Disease Detection — Project Explanation

> A comprehensive guide to understanding every aspect of this project: what it does, how it works, and how to reproduce it.

---

## 📌 1. Project Overview

### What does this project do?

This project builds a **Convolutional Neural Network (CNN)** that takes a photograph of a plant leaf as input and predicts which **disease** (if any) the plant is suffering from.

### Why does it matter?

- Plant diseases cause significant crop loss worldwide.
- Manual inspection by experts is slow and doesn't scale.
- A deep-learning model can classify diseases from a simple photograph in **milliseconds**, enabling early intervention.

### Key numbers

| Metric | Value |
|---|---|
| Number of classes | **38** |
| Image input size | **224 × 224 × 3** (RGB) |
| Model type | Sequential CNN |
| Training epochs | 10 |
| Framework | TensorFlow / Keras |

---

## 🛠️ 2. Tech Stack

| Library | Version (approx.) | Role |
|---|---|---|
| `Python` | 3.10+ | Language |
| `TensorFlow` / `Keras` | 2.x | CNN definition, training, inference |
| `NumPy` | 1.x | Array math, seeding |
| `Matplotlib` | 3.x | Plotting accuracy/loss curves and predictions |
| `PIL` (Pillow) | — | Loading and inspecting individual images |
| `zipfile` | stdlib | Extracting the downloaded dataset |
| `os`, `json` | stdlib | File system operations, Kaggle credentials |
| `kaggle` (pip) | — | Downloading the dataset from Kaggle |
| **Google Colab** | — | Cloud environment with free GPU |

---

## 📦 3. Dataset Details

### Source

The **PlantVillage** dataset, hosted on Kaggle:
- **URL:** <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset>
- **License:** CC-BY-NC-SA-4.0
- **Download size:** ~2 GB (zipped)

### What's inside?

The dataset contains images organized into **three folders**:

| Folder | Description |
|---|---|
| `color/` | Original RGB leaf photographs (**used in this project**) |
| `grayscale/` | Grayscale versions of the same images |
| `segmented/` | Background-removed versions |

Each folder contains **38 sub-folders**, one per class. The class names follow the pattern `Plant___Disease`, for example:

- `Tomato___Early_blight`
- `Potato___Late_blight`
- `Grape___healthy`
- `Apple___Cedar_apple_rust`
- `Corn_(maize)___Common_rust_`

Some classes represent **healthy** leaves (no disease), while others represent specific diseases.

### Image properties

- **Original dimensions:** 256 × 256 pixels, 3 channels (RGB)
- **Example class size:** `Grape___healthy` contains **423** images

---

## 🔧 4. Data Pipeline

The notebook uses Keras' `ImageDataGenerator` to feed images to the model:

### Step-by-step

1. **Download:** The Kaggle API is used to download the dataset programmatically.
2. **Extract:** The ZIP file is extracted using Python's `zipfile` module.
3. **Choose variant:** Only the `color/` folder is used (`base_dir = 'plantvillage dataset/color'`).
4. **Create generators:**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rescale pixel values from [0, 255] to [0, 1]
data_gen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2    # 80% train, 20% validation
)

# Training generator
train_data = data_gen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # resize all images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation generator
val_data = data_gen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
```

### Key parameters

| Parameter | Value | Why |
|---|---|---|
| `rescale` | `1/255` | Normalizes pixels to [0, 1] — helps gradient descent converge |
| `validation_split` | `0.2` | Reserves 20% of data for validation (no separate test set) |
| `target_size` | `(224, 224)` | Standard input size for many CNNs; resizes all images uniformly |
| `batch_size` | `32` | Processes 32 images per gradient update |
| `class_mode` | `'categorical'` | One-hot encodes the 38 class labels |

---

## 🧬 5. Model Architecture

The model is a **Sequential CNN** — a stack of layers where each layer's output flows directly into the next.

### Layer-by-layer breakdown

```
┌────────────────────────────────────────────────┐
│  Input: 224 × 224 × 3 (RGB image)              │
├────────────────────────────────────────────────┤
│  Conv2D(32, kernel=3×3, activation='relu')     │  → learns 32 low-level filters (edges, textures)
│  MaxPooling2D(pool_size=2×2)                   │  → halves spatial dimensions → 112 × 112
├────────────────────────────────────────────────┤
│  Conv2D(64, kernel=3×3, activation='relu')     │  → learns 64 mid-level features
│  MaxPooling2D(pool_size=2×2)                   │  → 56 × 56
├────────────────────────────────────────────────┤
│  Conv2D(128, kernel=3×3, activation='relu')    │  → learns 128 higher-level patterns
│  MaxPooling2D(pool_size=2×2)                   │  → 28 × 28
├────────────────────────────────────────────────┤
│  Conv2D(128, kernel=3×3, activation='relu')    │  → learns 128 more abstract features
│  MaxPooling2D(pool_size=2×2)                   │  → 14 × 14
├────────────────────────────────────────────────┤
│  Flatten()                                      │  → reshape to 1D vector (14 × 14 × 128 = 25,088)
│  Dense(512, activation='relu')                 │  → fully connected hidden layer
│  Dense(38, activation='softmax')               │  → output probabilities for 38 classes
└────────────────────────────────────────────────┘
```

### Why this architecture?

- **Conv2D layers** extract spatial features (edges → textures → shapes → disease patterns).
- **MaxPooling** reduces the spatial size, cutting computation and helping the model generalize.
- **Flatten + Dense** act as a classifier on top of the extracted features.
- **Softmax** turns raw scores into a probability distribution across 38 classes.

### Compilation

```python
model.compile(
    optimizer='adam',                # Adaptive learning rate optimizer
    loss='categorical_crossentropy', # Standard loss for multi-class classification
    metrics=['accuracy']
)
```

| Choice | Reason |
|---|---|
| **Adam** | Adapts learning rate per-parameter; fast convergence |
| **Categorical cross-entropy** | Measures how far the predicted probability distribution is from the true one-hot label |
| **Accuracy** | Easy-to-interpret metric for classification |

---

## 🏋️ 6. Training

The model is trained with:

```python
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```

- **10 epochs** means the model sees the entire training set 10 times.
- The `history` object stores per-epoch values of loss and accuracy for both training and validation sets.

### What to watch for

| Scenario | Meaning |
|---|---|
| Training accuracy ↑, validation accuracy ↑ | ✅ Model is learning and generalizing |
| Training accuracy ↑, validation accuracy ↓ | ⚠️ Overfitting — model memorizes training data |
| Both accuracies plateau early | Model may need more capacity or data augmentation |

---

## 📈 7. Evaluation & Visualization

### Accuracy & Loss Curves

The notebook plots training vs. validation accuracy and loss over epochs:

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
```

These plots help diagnose overfitting, underfitting, or healthy training.

### Prediction on Test Images

The model predicts on images from the validation set and displays them with:
- The **true label** (from the directory name)
- The **predicted label** (class with highest softmax probability)

This visual check is critical — it shows whether the model's decisions make intuitive sense.

---

## 🔁 8. How to Reproduce

### Option A — Google Colab (recommended)

1. Open the notebook in Google Colab (click the badge at the top of the `.ipynb` file).
2. Upload your `kaggle.json` file when prompted.
3. Run all cells top-to-bottom.
4. Training will use Colab's free GPU.

### Option B — Local machine

1. **Clone the repo:**
   ```bash
   git clone https://github.com/vishwanshv/Plant-Disease-Detection.git
   cd Plant-Disease-Detection
   ```
2. **Install dependencies:**
   ```bash
   pip install tensorflow numpy matplotlib pillow kaggle
   ```
3. **Set up Kaggle credentials:**
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<you>\.kaggle\` (Windows).
4. **Run the notebook:**
   ```bash
   jupyter notebook Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
   ```
5. Execute all cells sequentially.

---

## 📝 9. Code Walkthrough (Cell-by-Cell)

### Cell 1 — Seeding for Reproducibility

```python
import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.random.set_seed(0)
```

Sets random seeds for Python, NumPy, and TensorFlow so that results are **reproducible** — running the same code multiple times gives the same output.

---

### Cell 2 — Importing Dependencies

```python
import os, json
from zipfile import ZipFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
```

- `os`, `json` — file system and credential handling
- `ZipFile` — extracting the downloaded dataset
- `PIL.Image` — loading individual images for inspection
- `matplotlib` — visualization
- `ImageDataGenerator` — feeds batches of pre-processed images to the model
- `layers`, `models` — building the CNN

---

### Cell 3 — Installing & Configuring Kaggle API

```python
!pip install kaggle
kaggle_credentials = json.load(open("kaggle.json"))
os.environ['KAGGLE_USERNAME'] = kaggle_credentials["username"]
os.environ['KAGGLE_KEY'] = kaggle_credentials["key"]
```

Reads your Kaggle API token and sets it as environment variables so the CLI can authenticate.

---

### Cell 4 — Downloading the Dataset

```python
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
```

Downloads the ~2 GB PlantVillage ZIP file into the Colab working directory.

---

### Cell 5 — Extracting the Dataset

```python
with ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall()
```

Unzips into a folder called `plantvillage dataset/` with sub-folders `color/`, `grayscale/`, `segmented/`.

---

### Cell 6 — Exploring the Data

```python
print(os.listdir("plantvillage dataset"))         # ['segmented', 'color', 'grayscale']
print(len(os.listdir("plantvillage dataset/color")))  # 38 classes
```

Confirms the folder structure and class count.

---

### Cell 7 — Inspecting a Sample Image

```python
img = Image.open("plantvillage dataset/color/Grape___healthy/<filename>.JPG")
print(img.size)  # (256, 256)
```

Verifies image dimensions and visually inspects a sample leaf.

---

### Cell 8 — Setting Up Data Generators

Creates `ImageDataGenerator` with rescaling and a validation split, then builds `train_data` and `val_data` generators using `flow_from_directory`. See [Section 4](#-4-data-pipeline) for full details.

---

### Cell 9 — Building the CNN

Uses `models.Sequential` to stack Conv2D, MaxPooling, Flatten, and Dense layers. See [Section 5](#-5-model-architecture) for the full architecture diagram.

---

### Cell 10 — Compiling the Model

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Configures the optimizer, loss function, and evaluation metric.

---

### Cell 11 — Training the Model

```python
history = model.fit(train_data, validation_data=val_data, epochs=10)
```

Trains for 10 epochs, storing metrics in `history`.

---

### Cell 12 — Plotting Accuracy & Loss

Plots `history.history['accuracy']` and `history.history['val_accuracy']` (and the corresponding loss values) to visualize training progress.

---

### Cell 13 — Making Predictions

Takes sample images from the validation set, runs `model.predict()`, maps the predicted index to a class name, and displays the image alongside both the true and predicted labels.

---

## 💡 10. Key Concepts for Interviews

| Question | Answer |
|---|---|
| **Why CNN over a regular NN?** | CNNs exploit spatial structure in images; regular NNs would flatten the image and lose all spatial information |
| **Why rescale to 1/255?** | Neural networks train faster and more stably when inputs are in a small range like [0, 1] |
| **Why Adam optimizer?** | It adapts the learning rate for each parameter, combining the benefits of AdaGrad and RMSProp |
| **Why softmax in the output layer?** | Converts raw logits to a valid probability distribution over 38 classes |
| **Why categorical cross-entropy?** | It's the standard loss for multi-class classification with one-hot encoded labels |
| **What does MaxPooling do?** | Reduces spatial dimensions by taking the max in each pool window — this reduces computation Preserves the most prominent features |
| **How would you improve accuracy?** | Use transfer learning (e.g., ResNet-50), add data augmentation (rotations, flips), train for more epochs, use learning rate scheduling |
| **How would you deploy this?** | Export the model with `model.save()`, serve via Flask/Streamlit, or convert to TFLite for mobile |
