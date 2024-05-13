# 💡 Project Title
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
### Minimum Viable Product (MVP)
Our automated analysis model is designed to process video inputs by initially segmenting them into individual frames. Each frame undergoes preprocessing steps, which include binary labeling, augmenting the frames, as well as adding noise reduction techniques to enhance image clarity.
These preprocessed frames are then analyzed using a Vision Transformer (ViT) neural network, which extracts high-level features from each image. These feature vectors capture important patterns and characteristics present in the frames.
The extracted feature vectors are then inputted into a k-Nearest Neighbors (KNN) classifier for supervised feature vector detection. This detector categorizes each frame based on the learned patterns, facilitating the identification of potential issues or defects within the video footage.
Our MVP successfully achieves the task of identifying and returning timestamps of frames that are not classified as "clear," indicating the presence of anomalies or defects. During initial testing with a limited dataset, we achieved an impressive 99% accuracy rate. However, it's essential to note that this performance may not fully reflect real-world scenarios due to the limited amount of training data available at that time.
Moving forward, with the receipt of a larger dataset from IPIPE company, we have an opportunity to further enhance the performance and robustness of our model.

## ⚡ Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### 🧱 Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Jupyter Notebook or Google Colab
- `transformers` library
- `torch` library
- `scikit-learn` library
- `Pillow` library

### 🏗️ Installing
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/project-repo.git
    cd project-repo
    ```

2. **Install the Required Libraries:**
    ```bash
    pip install transformers torch scikit-learn Pillow
    ```

3. **Mount Google Drive (if using Colab):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4. **Load the Pre-trained Model:**
    ```python
    from transformers import ViTFeatureExtractor, ViTModel

    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    ```

### 🧪 Testing
To test the project, run the Jupyter Notebook cells. Here's an example of running a simple test:

1. **Load an Image and Apply Transformations:**
    ```python
    from PIL import Image
    import torchvision.transforms as transforms

    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])

    image = Image.open('path/to/image.jpg')
    transformed_image = data_transforms(image)
    ```

2. **Make a Prediction:**
    ```python
    with torch.no_grad():
        inputs = feature_extractor(images=transformed_image, return_tensors="pt")
        outputs = model(**inputs)
        # Add your classifier logic here
    ```

## 🚀 Deployment
To deploy the model, follow the instructions for setting up a server or use a platform like AWS, GCP, or Azure for hosting.

## ⚙️ Built With
- [Transformers](https://github.com/huggingface/transformers) - Pre-trained models and feature extractors.
- [PyTorch](https://pytorch.org/) - Deep learning framework.
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library.
- [Pillow](https://python-pillow.org/) - Image processing library.

## 🙏 Acknowledgments
- Thanks to everyone who contributed to the libraries and tools used in this project.
