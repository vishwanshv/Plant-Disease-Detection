# 🌿 Plant Disease Detection using CNN

## 📌 Overview

This project leverages **Convolutional Neural Networks (CNNs)** to detect and classify plant diseases from leaf images. By automating disease identification, it enables farmers and researchers to take timely action to protect crops and improve yield.

The model is trained on the **PlantVillage** dataset and learns to distinguish between **healthy leaves** and those affected by **38 different plant disease classes** across multiple crop species.

---

## 🚀 Features

- 🧠 Image classification using Deep Learning (CNN)
- 🖼️ Image preprocessing pipeline:
  - Resizing to 224×224
  - Normalization (rescaling pixel values to [0, 1])
  - 80/20 train–validation split
- 📊 Model training and evaluation in Google Colab
- 📈 Visualization of:
  - Accuracy curves (training & validation)
  - Loss curves (training & validation)
  - Model predictions on test images
- 💾 Exportable trained model for deployment

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **TensorFlow / Keras** | Deep learning framework |
| **NumPy** | Numerical computation |
| **Matplotlib** | Plotting & visualization |
| **PIL (Pillow)** | Image loading & inspection |
| **Google Colab** | Training & experimentation environment |

---

## 📂 Project Structure

```
Plant-Disease-Detection/
├── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb   # Main notebook
└── README.md                                             # This file
```

---

## 📦 Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **License:** CC-BY-NC-SA-4.0
- **Classes:** 38 (plant–disease combinations, e.g., `Tomato___Early_blight`, `Grape___healthy`)
- **Image variants:** Color, Grayscale, Segmented (this project uses **color** images)
- **Original image size:** 256 × 256 × 3
- **Size:** ~2 GB (zipped)

---

## 🧬 Model Architecture

The CNN is built using the Keras `Sequential` API:

```
Input (224×224×3)
  → Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
  → Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
  → Flatten
  → Dense(512, ReLU)
  → Dense(38, Softmax)        ← output (one per class)
```

- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Epochs:** 10

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/vishwanshv/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2️⃣ Open the Notebook

Open the notebook in:
- **Google Colab** (Recommended) — click the "Open in Colab" badge at the top of the notebook
- **OR** Jupyter Notebook locally:

```bash
jupyter notebook Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
```

### 3️⃣ Upload Dataset

- Upload your `kaggle.json` credentials file, **OR**
- Link the Kaggle dataset directly using the Kaggle API:

```bash
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
```

### 4️⃣ Train the Model

Run all cells sequentially to:
1. Download and extract the dataset
2. Preprocess images using `ImageDataGenerator`
3. Build and compile the CNN
4. Train for 10 epochs
5. Visualize training/validation accuracy & loss curves

### 5️⃣ Make Predictions

- Load the trained model
- Provide new leaf images
- Get disease classification results with predicted class labels

---

## 📊 Results

- ✅ Achieved high accuracy in classifying multiple plant diseases
- ✅ Smooth convergence in loss and accuracy curves
- ✅ Strong generalization on unseen leaf images
- ✅ Sample predictions visualized in the notebook

---

## 🌍 Applications

- 🌱 Early detection of plant diseases
- 🚜 Precision agriculture support
- 🔬 Crop health monitoring research
- 📱 Potential integration into mobile applications for farmers

---

## 🔮 Future Improvements

- 🌐 Deploy as a web application (Streamlit / Flask)
- 📲 Convert to TensorFlow Lite for mobile deployment
- 🧪 Improve accuracy using transfer learning (ResNet, EfficientNet)
- 📁 Expand dataset for more plant species

---

## 📜 Acknowledgements

- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) by Abdallah Ali (CC-BY-NC-SA-4.0)
- **Framework:** [TensorFlow / Keras](https://www.tensorflow.org/)
- **Environment:** [Google Colab](https://colab.research.google.com/)
