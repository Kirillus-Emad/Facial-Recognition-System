# Facial-Recognition-System

A fully integrated **Facial Recognition System** that includes real-time face detection using **YOLOv8**, face embedding extraction using **ArcFace**, attendance tracking, PostgreSQL database support, dynamic dashboards, reporting tools, and complete MLOps integration (MLflow, retraining triggers, monitoring).

---

## 🖥️ System Requirements

### **Hardware Requirements**
| Component | Minimum Requirement |
|----------|----------------------|
| **CPU** | Intel Core i5 (4 cores) or AMD Ryzen 5 |
| **RAM** | 3 GB |
| **GPU** | Not required *(Optional: CUDA-enabled GPU for faster processing)* |
| **Storage** | 1 GB free space |
| **Camera** | USB webcam (optional for real-time recognition) |

> **Note:** GPU improves performance significantly if PyTorch with **CUDA 11.3** is installed.

---

## 📦 Software Requirements

### **Required Software**
- **Python 3.10**
- **pip** (Python package manager)
- **Modern web browser** (Chrome, Firefox, Edge)

### **Optional Software**
- **CUDA Toolkit 11.3** (for YOLO & ArcFace GPU acceleration)
- **cuDNN 8.1+**

---

## ⚙️ Installation & Setup Instructions

Below are the complete installation steps for preparing the environment and downloading required models.

### **1. Create an Anaconda Environment (Recommended)**
This avoids dependency conflicts:
```bash
conda create -n face_env python=3.10
conda activate face_env
```

---

### **2. Install the Requirements File**
Install all dependencies for the system:
```bash
pip install -r requirements.txt
```

---

### **3. Restrict NumPy to Compatible Version**
This ensures compatibility with the detection and embedding models:
```bash
pip install numpy==1.23.5
```
> You can safely ignore dependency warnings—everything works correctly.

---

### **4. Download YOLOv8 Face Detection Model**
Download any YOLOv8 model you prefer (e.g., nano, small).  
The **small** model is recommended for better accuracy:
🔗 https://drive.google.com/drive/folders/1QR56w-sNWvDBUW1pYLUry-qosWo79u6L?usp=drive_link

Place the downloaded model inside the appropriate `models/` folder.

---

### **5. Download Face Embedding Extractor (ArcFace Model)**
This model is responsible for generating embeddings to compare identities:
🔗 https://drive.google.com/file/d/14r9fW_ibx2fZTDJKIzPj-Bwjrb-VLr6y/view?usp=drive_link

Store the file in your `models/` directory.

---

### **6. (Optional) PostgreSQL Database Integration**
If you have your own PostgreSQL database:

1. Open the `.env` file  
2. Modify the connection string:
```
DataBaseUrl=postgresql://username:password@host:port/database
```

This allows storing face records, attendance logs, and user data.

---

## 🧑‍💻 User Manual

Follow these steps to run the system after setup.

---

### **1. Activate Your Environment**
```bash
conda activate face_env
```

---

### **2. Navigate to the Project Directory**
Go to the folder containing:
```
start_system.py
```

---

### **3. Start the System**
Run:
```bash
python start_system.py
```

Wait until you see:
```
Application start
```

---

### **4. Open the Web Interface**
Go to:
👉 **http://127.0.0.1:8000/**

This is the main dashboard where all functionalities are available.

---

## 🌐 Website Functionalities

### **You can:**
✔ Upload a video to detect and recognize faces  
✔ Use your webcam for real-time recognition  
✔ Register a new face into the system  
✔ Delete any face ID from the database  
✔ Download:
- **Monthly Report**  
- **Detailed Report**

---

## 📊 Dynamic Recognition Dashboard

When you upload a video or activate webcam live detection, you’ll see:

### **1. Total Faces**
- Number of all faces detected so far.

---

### **2. Present (Known Persons)**
Clicking the label shows:
- The person’s cropped face
- Their face embedding distance score  

**Interpretation of score:**
- **Score < 1.04 → Strong match**  
- Lower score = higher confidence.

---

### **3. Unknown Faces**
Clicking shows:
- Unknown person’s cropped face  
- Their embedding distance score  

**Interpretation of score:**
- **Score > 1.04 → Confident Unknown**

---

### **4. Download Dashboard Statistics**
Generates a `.txt` file containing:
- Video file name  
- Start & end time  
- Total faces detected  
- Known persons detected  
- Unknown persons detected  
- All IDs that appeared in the session  

---

### **5. Stop Processing**
You can stop webcam/video processing at any time using **"Stop Processing"**.

---

## 📑 Reports

### **Monthly Report**
Shows how many times a person appeared in the current month.

### **Detailed Report**
Includes:
- Total appearances since registration  
- All specific dates attended  

---

## 🤖 MLOps Tools & Monitoring

The system includes powerful MLOps integration.

### **MLflow Dashboard**
Track models, metrics, and experiments:
```
http://localhost:5000
```

### **MLOps Status**
Check current ML pipeline status:
```
http://localhost:8000/mlops/status
```

### **Trigger Model Retraining**
Send a POST request to retrain models:
```
http://localhost:8000/mlops/retrain
```

---

## ✔️ Summary

With this system, you can:
- Detect faces in real-time or from videos  
- Identify known persons using ArcFace embeddings  
- Track attendance automatically  
- Manage faces in your PostgreSQL database  
- Monitor and retrain ML models  
- Export attendance and dashboard reports  
- View live statistics for every running session  

All packaged in a single, easy-to-use interface.

---
