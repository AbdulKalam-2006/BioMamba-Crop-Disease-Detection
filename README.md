# BioMamba: Bio-Inspired Crop Disease Detection

### Bio-Inspired Crop Disease Detection via Vascular Scanning and State-Space Diffusion

BioMamba is an AI-powered crop disease detection system that explores bio-inspired vascular scanning and State Space Modeling to identify diseases in crop leaves.

The proposed system combines **MobileNetV3**, a **Vascular Cross-Scanning Engine**, and a **Mamba State Space Model** to analyze spatial features and disease patterns in plant leaf images.

The project aims to support intelligent agriculture through efficient, automated and accessible crop disease identification.

---

## 1. Project Overview

Traditional crop disease identification often requires manual observation and agricultural expertise. BioMamba explores an AI-based approach to automate the identification process using deep learning and bio-inspired feature analysis.

Unlike conventional approaches that primarily focus on local image features, BioMamba introduces directional vascular scanning to analyze spatial relationships across leaf regions.

### Key Objectives

- Develop an AI-based crop disease detection system.
- Explore bio-inspired vascular feature extraction.
- Integrate MobileNetV3 for efficient feature extraction.
- Apply Mamba State Space Modeling for spatial sequence analysis.
- Provide disease classification and confidence scores.
- Support offline inference and agricultural applications.

---

## 2. Key Features

- **AI-Based Disease Detection:** Identifies crop diseases using leaf images.
- **MobileNetV3 Feature Extraction:** Extracts meaningful visual features efficiently.
- **Vascular Cross-Scanning:** Analyzes horizontal and vertical spatial patterns.
- **Mamba State Space Model:** Models long-range spatial relationships.
- **Disease Classification:** Predicts the corresponding disease category.
- **Confidence Analysis:** Displays prediction probabilities.
- **Clinical Treatment Advisory:** Provides treatment-related guidance for supported predictions.
- **Out-of-Distribution Detection:** Helps identify unsupported or uncertain inputs.
- **Streamlit Interface:** Provides an interactive application.
- **FastAPI Backend:** Supports API-based inference.
- **Offline Inference:** Supports deployment-oriented model export workflows.

---

## 3. Supported Crops

BioMamba is designed for disease identification across five major crop categories.

| Crop | Crop Category |
|---|---|
| Corn | Cereal |
| Potato | Tuber |
| Rice | Cereal |
| Sugarcane | Commercial Crop |
| Wheat | Cereal |

The classification system covers healthy and disease-related categories represented in the project's training configuration.

---

## 4. System Architecture

The BioMamba architecture consists of the following processing stages:

```text
             INPUT LEAF IMAGE
                    |
                    v
           IMAGE PREPROCESSING
           Resize and Normalize
                    |
                    v
          MobileNetV3 BACKBONE
            Feature Extraction
                    |
                    v
         VASCULAR CROSS-SCANNING
                    |
             +------+------+
             |             |
             v             v
       Horizontal      Vertical
        Scanning       Scanning
             |             |
             +------+------+
                    |
                    v
           FEATURE SEQUENCES
                    |
                    v
          MAMBA STATE SPACE
                MODEL
                    |
                    v
          DISEASE CLASSIFIER
                    |
                    v
         CONFIDENCE ANALYSIS
                    |
                    v
       DISEASE PREDICTION
       AND TREATMENT ADVISORY
```

<img width="846" height="2588" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/96ac7262-5719-43ea-894c-6c644a52e7c2" />


### Architecture Components

**1. Image Preprocessing**

- Accepts leaf images.
- Resizes images to the required input dimensions.
- Applies normalization and preprocessing operations.

**2. MobileNetV3 Feature Extraction**

- Extracts spatial features from the input image.
- Uses an efficient convolutional neural network backbone.
- Produces feature representations for subsequent processing.

**3. Vascular Cross-Scanning Engine**

- Performs horizontal and vertical feature scanning.
- Generates directional feature sequences.
- Explores spatial relationships inspired by plant vascular structures.

**4. Mamba State Space Model**

- Processes the extracted feature sequences.
- Models long-range dependencies.
- Provides computationally efficient sequence modeling.

**5. Disease Classification**

- Produces disease classification logits.
- Calculates class probabilities.
- Identifies the predicted disease category.

**6. Confidence and Advisory Module**

- Displays prediction probabilities.
- Handles uncertain predictions.
- Provides treatment-related advisory information.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Feature Extraction | MobileNetV3 |
| Sequence Modeling | Mamba State Space Model |
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Frontend | Streamlit |
| Backend | FastAPI |
| API Server | Uvicorn |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Offline Inference | ONNX Runtime |
| Development Environment | Visual Studio Code |

---

## 6. Project Structure

```text
BioMamba-Crop-Disease-Detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── dataset/
    │
    ├── .gitignore
    ├── app.py
    ├── api.py
    ├── dataset_loader.py
    ├── export_mobile.py
    ├── export_offline.py
    ├── generate_plots.py
    ├── loss.py
    ├── model.py
    ├── test_offline_inference.py
    └── train.py
```

### Source Code Description

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `api.py` | FastAPI backend |
| `model.py` | BioMamba model architecture |
| `dataset_loader.py` | Dataset loading and preprocessing |
| `train.py` | Model training |
| `loss.py` | Bio-Diffusion loss implementation |
| `export_mobile.py` | Mobile model export |
| `export_offline.py` | Offline model export |
| `generate_plots.py` | Training and evaluation visualizations |
| `test_offline_inference.py` | ONNX inference testing |

---

## 7. Installation and Setup

### Prerequisites

- Python with compatible package support
- Git
- Visual Studio Code (recommended)
- Windows, Linux or macOS

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbdulKalam-2006/BioMamba-Crop-Disease-Detection.git
```

### Step 2: Navigate to the Project Directory

```bash
cd BioMamba-Crop-Disease-Detection
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

**Note:** PyTorch and Torchvision installations must be compatible with the selected Python version and operating system. GPU users may need a platform-specific PyTorch installation.

---

## 8. Running the Application

### A. Launch Streamlit Frontend

From the project root directory:

```bash
streamlit run dataset/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

### B. Launch FastAPI Backend

```bash
uvicorn api:app --app-dir dataset --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### C. Model Training

After preparing the required training dataset:

```bash
python dataset/train.py
```

### D. Offline Inference Testing

After preparing the required exported ONNX model and its associated files:

```bash
python dataset/test_offline_inference.py
```

---

## 9. Dataset and Model Weights

**Important:** This public repository contains the project source code and documentation. Large datasets and trained model weights are intentionally excluded.

The following directories and files are not included:

- Training datasets
- Model checkpoints
- `.pth` files
- `.pt` files
- `.onnx` model files
- Virtual environment files

To run model inference, provide the appropriate trained checkpoint and any required model metadata in the expected local model directory.

To reproduce training, prepare the dataset in the directory structure expected by `dataset_loader.py`.

---

## 10. Applications

- Smart Agriculture
- Automated Crop Disease Identification
- Agricultural Decision Support Systems
- Precision Farming
- AI-Based Plant Health Monitoring
- Edge AI and Mobile Agricultural Applications

---

## 11. Future Enhancements

- Android application deployment.
- Optimized edge-device inference.
- Support for additional crop varieties.
- Multilingual agricultural assistance.
- Integration with smart farming systems.
- Drone-based crop monitoring.
- Agricultural advisory chatbot.
- Integration with automated spraying technologies.

---

## 12. Research Significance

BioMamba explores the combination of bio-inspired spatial analysis and state-space sequence modeling for crop disease detection.

The project investigates how directional feature scanning can complement conventional convolutional feature extraction and how Mamba-based modeling can process spatial feature sequences efficiently.

The proposed approach is intended to contribute to research in AI-driven precision agriculture and intelligent crop health monitoring.

---

## 13. Project Information

| Field | Details |
|---|---|
| Project Name | BioMamba |
| Full Title | Bio-Inspired Crop Disease Detection |
| Domain | Artificial Intelligence and Machine Learning |
| Subdomain | Deep Learning and Computer Vision |
| Application Area | Smart Agriculture |
| Frontend | Streamlit |
| Backend | FastAPI |
| Repository | GitHub |

---

## 14. Repository

**GitHub:** [BioMamba-Crop-Disease-Detection](https://github.com/AbdulKalam-2006/BioMamba-Crop-Disease-Detection)

---

## 15. License

A license has not yet been selected for this repository. Unless a license is added, standard copyright restrictions apply to reuse and redistribution.

---

**BioMamba – Exploring Bio-Inspired Intelligence for Smarter Agriculture.**
