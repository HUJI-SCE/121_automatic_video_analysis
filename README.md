# 💡 Auto-Analysis of iPipe Video’s Data 
Automated Defect Classification in Water Pipelines using Vision Transformers

<!-- cool project cover image -->
![Project Cover Image](/media/back.png)

## Table of Contents
- [The Team](#the-team)
- [Project Description](#project-description)
- [Method](#method)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installing](#installing)
- [Testing](#testing)
- [Deployment](#deployment)
- [Built With](#built-with)
- [Acknowledgments](#acknowledgments)

## 👥 The Team 
**Team Members**
- [Ibrahem Alyan](ibrahem.alyan@mail.huji.ac.il)
- [Mustafa Shouman](name@mail.huji.ac.il)
- [Riham Aladam](name@mail.huji.ac.il)

**Supervisor**
- [Eliahu Horwitz](wwww.link_to_lab.com)

## 📚 Project Description
Our solution aims to automate the classification of defects in water pipelines using IPIPE’s video data. We employ a Vision Transformer (ViT) Neural Network to analyze the data and extract vectors of image features. These feature vectors are then classified using a KNN classifier. This approach leverages the power of deep learning and machine learning techniques to accurately identify and classify anomalies in pipeline videos, contributing to improved maintenance and operational efficiency.

## ⚙️ Method
Our automated analysis model is designed to process video inputs by initially segmenting them into individual frames. Each frame undergoes preprocessing steps, which include binary labeling, augmenting the frames, as well as adding noise reduction techniques to enhance image clarity.
These preprocessed frames are then analyzed using a Vision Transformer (ViT) neural network, which extracts high-level features from each image. These feature vectors capture important patterns and characteristics present in the frames.
The extracted feature vectors are then inputted into a k-Nearest Neighbors (KNN) classifier for supervised feature vector detection. This detector categorizes each frame based on the learned patterns, facilitating the identification of potential issues or defects within the video footage.


## ⚡ Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.


### 🏗️ Installing
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/project-repo.git
    cd project-repo
    ```

2. **Install the Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3. **run streamlit UI:**
   ```bash
    streamlit run app.py
    ```
   


## ⚙️ Built With
- [Transformers](https://github.com/huggingface/transformers) - Pre-trained models and feature extractors.
- [PyTorch](https://pytorch.org/) - Deep learning framework.
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library.
- [Pillow](https://python-pillow.org/) - Image processing library.
