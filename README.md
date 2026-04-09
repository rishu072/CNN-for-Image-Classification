# 🚀 CNN Image Classification using CIFAR-10

## 📌 Project Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset into 10 categories such as airplane, car, dog, cat, etc.

The goal of this assignment is to understand CNN architecture, training process, and evaluation of image classification models.

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## 📂 Project Structure

```
CNN-Assignment/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│
├── main.py
├── requirements.txt
├── README.md
├── report.pdf
```

---

## 📦 Dataset

* Dataset: CIFAR-10
* Total Images: 60,000
* Classes: 10
* Loaded using `torchvision.datasets.CIFAR10`

---

## 🧠 Model Architecture

* Conv Layer 1 → ReLU → MaxPool
* Conv Layer 2 → ReLU → MaxPool
* Fully Connected Layer
* Output Layer (10 classes)

---

## 🔁 Training Details

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 0.001
* Epochs: 3 (for fast training)

---

## 📊 Results

* Accuracy: ~40–60% (basic model)
* Loss decreases during training
* Model successfully classifies CIFAR-10 images

---

## 🚀 How to Run

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

### Step 2: Run the project

```
python main.py
```

---

## 📈 Output

* Training Loss printed per epoch
* Final Accuracy
* Loss Graph

---

## 📌 Conclusion

The CNN model was successfully implemented and trained on the CIFAR-10 dataset. The project demonstrates the fundamentals of convolutional neural networks and image classification.

---

## 🙌 Author

Student Assignment Submission
