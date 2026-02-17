# 🌿 Plant Disease Detection using CNN

## 📌 Overview

This project leverages **Convolutional Neural Networks (CNNs)** to detect and classify plant diseases from leaf images. By automating disease identification, it enables farmers and researchers to take timely action to protect crops and improve yield.

The model is trained on a dataset of plant leaf images and learns to distinguish between **healthy leaves** and those affected by common plant diseases.

---

## 🚀 Features

- 🧠 Image classification using Deep Learning (CNN)
- 🖼️ Image preprocessing pipeline:
  - Resizing
  - Normalization
  - Data augmentation
- 📊 Model training and evaluation in Google Colab
- 📈 Visualization of:
  - Accuracy curves
  - Loss curves
  - Model predictions
- 💾 Exportable trained model for deployment

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Google Colab** (Training & experimentation)

---

## 📂 Project Structure

Plant-Disease-Detection/
│
├── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb # Main notebook
├── data/ # Dataset (leaf images)
├── models/ # Saved trained models
├── results/ # Accuracy/loss plots, predictions
└── README.md # Project documentation


---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Plant-Disease-Detection.git
cd Plant-Disease-Detection
2️⃣ Open the Notebook
Open the notebook in:

Google Colab (Recommended)
OR

Jupyter Notebook

jupyter notebook Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
3️⃣ Upload Dataset
Upload your dataset manually
OR

Link your Kaggle dataset (if applicable)

4️⃣ Train the Model
Run all cells sequentially to train and evaluate the CNN model.

5️⃣ Make Predictions
Load the trained model

Provide new leaf images

Get disease classification results

📊 Results
Achieved high accuracy in classifying multiple plant diseases

Smooth convergence in loss and accuracy curves

Strong generalization on unseen leaf images

Sample predictions visualized in the notebook

🌍 Applications
Early detection of plant diseases

Precision agriculture support

Crop health monitoring research

Potential integration into mobile applications for farmers

🔮 Future Improvements
Deploy as a web application (Streamlit / Flask)

Convert to TensorFlow Lite for mobile deployment

Improve accuracy using transfer learning (ResNet, EfficientNet)

Expand dataset for more plant species
