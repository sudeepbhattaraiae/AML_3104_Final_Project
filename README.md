# Plant Leaf Diseases Prediction




## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Contributors](#contributors)

## Project Description
This project aims to detect plant diseases from leaf images using Convolutional Neural Networks (CNNs). By leveraging the PlantVillage dataset, we have developed a machine learning pipeline to preprocess the data, train deep learning models, and deploy a user-friendly web application for real-time plant disease diagnosis.

---

## **Dataset Information**
The **PlantVillage dataset** contains over 50,000 images of healthy and diseased leaves across multiple plant species. It is an open-source dataset designed to help in diagnosing plant diseases. Each image is labeled with the crop type and the specific disease.

- **Source**: [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Number of Classes**: 38 (e.g., healthy, bacterial spot, leaf mold, etc.)
- **Data Distribution**:
  - Images are categorized by crop and disease type.
  - Balanced across classes to ensure robust model training.

---

## Installation
This project requires Python 3.6+ and the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

To run this project, download the project files and run the Jupyter Notebook.

## Usage
After installing the necessary libraries, open the Jupyter Notebook. You can view the code and output for each cell and run each cell individually. To run all cells at once, click Cell -> Run All in the menu.

## **Dataset Information**
The **PlantVillage dataset** contains over 50,000 images of healthy and diseased leaves across multiple plant species. It is an open-source dataset designed to help in diagnosing plant diseases. Each image is labeled with the crop type and the specific disease.

- **Source**: [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Number of Classes**: 38 (e.g., healthy, bacterial spot, leaf mold, etc.)
- **Data Distribution**:
  - Images are categorized by crop and disease type.
  - Balanced across classes to ensure robust model training.

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Resized all images to 128x128 pixels.
   - Applied normalization to scale pixel values.
   - Performed data augmentation (rotation, flipping, brightness adjustment) to increase model robustness.
   
2. **Exploratory Data Analysis**:
   - Visualized class distributions and example images.
   - Examined data quality and diversity across classes.

3. **Model Training**:
   - Developed a CNN model with transfer learning (using pre-trained architectures like ResNet and VGG).
   - Split dataset: 70% training, 20% validation, 10% testing.
   - Used techniques like dropout and batch normalization to prevent overfitting.

4. **Model Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score.
   - Confusion matrix to analyze class-wise performance.

5. **Web Application**:
   - Built a Streamlit web app where users can upload leaf images to detect diseases.
   - Integrated multiple model options for comparison.

---

## **Model Architecture and Selection**
We experimented with various CNN architectures:
1. **Custom CNN**:
   - Achieved ~95% validation accuracy.
   
2. **Xception**:
   - Achieved 93% validation accuracy.


---
---

## **Installation**
### **Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- Libraries: TensorFlow, PyTorch, NumPy, Pandas, Matplotlib, OpenCV, Streamlit

### **Setup Instructions**
1. Clone this repository:
   ```bash
   git clone https://github.com/sudeepbhattaraiae/AML_3104_Final_Project.git
   cd plant-disease-detection

## **Installation**

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

## **Run the Training Script**
```bash
python train.py
```

## **Usage**

### **Run the Web App**
1. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a leaf image and view the predicted disease along with confidence scores.

### **Interactive Features**
- Compare predictions from different models.
- View accuracy metrics and model descriptions.

---

## Contributors
We are a group of dedicated students working collaboratively on the Plant Leaf diseases Prediction project. Our diverse backgrounds and shared interest in machine learning have brought us together to develop this model. Each member has contributed significantly to various aspects of the project, from data preprocessing and feature engineering to model training and user interface design. We are committed to providing a valuable tool for sellers and buyers in the used car market, helping them make informed decisions about the Plant leaf Diseases. Below is a list of our group members:

| S/N | Full Name | Student ID |
| --- | --------- | ---------- |
| 1 | Bijay Adhikari | C0883819 |
| 2 | Shishir Dhakal | C0913605 |
| 3 | Meenu Sharma | C0908452 |
| 4 | Sona Thapa | C0909929 |
| 5 | Saurav Parajuli| C0905417 |
| 6 | Sudeep Bhattarai | C0905601 |

We appreciate your interest in our project and welcome any feedback or questions.
