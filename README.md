# 🖼️ Natural Scene Classification using Transfer Learning (ResNet-50 Fine-Tuning on NaSC-TG2 Dataset)

This repository presents a deep learning pipeline for Natural Scene Classification using Transfer Learning with a pre-trained ResNet-50 model. The model is fine-tuned on the NaSC-TG2 dataset, a benchmark dataset containing multiple natural scene categories.
This project demonstrates expertise in Computer Vision, Transfer Learning, and Deep Learning Model Optimization, making it highly relevant for academic and research applications (e.g., PhD applications).

📌 Project Overview

Natural scene classification is a fundamental problem in computer vision, with applications in:

Remote sensing

Autonomous driving

Robot navigation

Environmental monitoring

Image retrieval systems

This project fine-tunes ResNet-50, pre-trained on ImageNet, on the NaSC-TG2 dataset, achieving high accuracy through:

Transfer learning

Data augmentation

Feature extraction + fine-tuning


🧠 Key Features

✔ Fine-tuned ResNet-50 (ImageNet pre-trained)
✔ Full training pipeline (augmentation → training → evaluation)
✔ Early stopping and learning rate scheduling
✔ Classification report & confusion matrix
✔ Training curves (accuracy & loss plots)
✔ Model testing on custom images
✔ Fully reproducible PyTorch implementation

📁 Repository Structure
│── data/
│   ├── train/
│   ├── val/
│   └── test/
│
│── src/
│   ├── dataset.py
│   ├── train.py
│   ├── eval.py
│   └── utils.py
│
│── models/
│   └── resnet50_finetuned.pth
│
│── notebooks/
│   └── training_experiments.ipynb
│
│── results/
│   ├── confusion_matrix.png
│   ├── training_curve.png
│   └── sample_predictions.png
│
└── README.md

🗂️ Dataset: NaSC-TG2 (Natural Scene Classification)

The NaSC-TG2 dataset contains diverse natural scene categories such as:

Beaches

Forest

Buildings

Mountains

Waterfalls

Streets

Ice/Snow

Farmland

Deserts

and more…

Dataset Characteristics
Property	Description
Total Classes	10 natural scene categories
Image Format	RGB
Resolution	~256×256 (varies)
Train/Val/Test Split	You may define custom splits
🔧 Methodology
1. Data Preprocessing

Resizing to 224×224

Normalization using ImageNet mean/std

Augmentation:

Random rotation

Random horizontal flip

Color jitter

Random crop

2. Model Architecture

Using ResNet-50, pre-trained on ImageNet.

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False   # Freeze backbone

# Replace final layer
model.fc = nn.Linear(2048, num_classes)


Then unfreeze last few layers for fine-tuning.

3. Training Setup
Setting	Value
Optimizer	Adam
Learning Rate	1e-4 (fine-tuning), 1e-3 (classifier)
Loss Function	Cross-Entropy
Epochs	20–30
Scheduler	StepLR / CosineAnnealingLR
Batch Size	32
4. Evaluation Metrics

Top-1 accuracy

Confusion matrix

Precision, Recall, F1-score

Per-class accuracy

# 📊 Results
Performance (Sample Values – Customize with Your Results)
Metric	Value
Training Accuracy	98.4%
Validation Accuracy	94.7%
Test Accuracy	93.2%
F1 Score	0.935
Confusion Matrix

(Add actual image)


Training Curves

Sample Predictions

# ▶️ How to Run
1. Clone the repository
git clone https://github.com/your-username/natural-scene-classification.git
cd natural-scene-classification

2. Install dependencies
pip install -r requirements.txt

3. Train the model
python src/train.py

4. Evaluate
python src/eval.py

🧪 Inference on Custom Images
from PIL import Image
import torch
from torchvision import transforms

img = Image.open("sample.jpg")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

tensor = transform(img).unsqueeze(0)
pred = model(tensor)

🚀 Future Improvements

You may include plans such as:

Training ViT, EfficientNet, ConvNeXt for comparison

Multi-domain generalization experiments

Feature visualization (Grad-CAM)

Deployment using ONNX/TensorRT

Using larger remote sensing datasets

These improvements show research capability — very useful for PhD applications.

# 📚 Related Research

Kaiming He et al., Deep Residual Learning for Image Recognition, CVPR 2016

Transfer Learning for Natural Scene Classification

Deep Learning in Remote Sensing

# 👤 Author

Muhammad Akhtar

Research Assistant, Northwestern Polytechnical University

LinkedIn: https://www.linkedin.com/in/engr-akhtar-malik/

GitHub: https://github.com/Mohammad-Akhtar-Awan
