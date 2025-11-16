# 🚗 Vehicle Damage Detection using Deep Learning

A deep learning–powered application that detects vehicle damage by classifying car images into multiple categories.
This project uses PyTorch, ResNet50, Optuna hyperparameter tuning, and a Streamlit web app for deployment.

### **⚙️ Tech Stack Used**

- Python , PyTorch 

- Torchvision (ResNet50) → Pretrained model for transfer learning

- Optuna → Hyperparameter tuning for best accuracy

- Streamlit → Frontend UI for users to upload images

- FastAPI → Backend API for inference logic

- Streamlit Cloud → Deployment platform

### **📌 Problem Statement**

The objective of this project is to detect and classify car damage from uploaded vehicle images. The model identifies the type of damage by classifying each image into one of the following six categories:
 
--> Front Normal,  Front Breakage,  Front Crushed,  Rear Normal,  Rear Breakage  and  Rear Crushed

### **1️⃣ Hyperparameter Tuning (Optuna)**

Optuna was used to find the best learning rate, dropout, and optimizer settings.
Validation accuracy was used as the main metric.

### **2️⃣ Model Evaluation**

Performance measured using:
 Accuracy, Precision, Recall, F1-Score, Confusion Matrix

**ResNet50** showed the most stable and high-quality predictions, so it was chosen as the final model.

### **3️⃣ Deployment**

- **Frontend:** Built using Streamlit for image upload & prediction display

- **Backend:** Prediction logic served using FastAPI

- **Hosting:** Fully deployed on Streamlit Cloud, accessible online

🔗 Live App: https://dlcardamagedetection-ygzrie9ufw6qv48pntcjbh.streamlit.app/
