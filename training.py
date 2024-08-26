import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np


class PipeTrainer:
    """
    A class to train classifiers using features extracted from a Vision Transformer (ViT) model.

    Attributes:
        train_dataset (Dataset): The dataset used for training.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (torch.device): The device to use for computation (GPU or CPU).
        feature_extractor (ViTImageProcessor): The feature extractor (preprocessor) from the ViT model.
        model (ViTModel): The Vision Transformer (ViT) model used for feature extraction.
    """

    def __init__(self, train_dataset, batch_size=32):
        """
        Initializes the PipeTrainer with a training dataset and optional batch size.

        Args:
            train_dataset (Dataset): The dataset to be used for training.
            batch_size (int): The number of samples per batch to load. Default is 32.
        """
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name, return_tensors='pt')
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def extract_features(self, concatenate_layers=True):
        """
        Extracts features from the training dataset using the Vision Transformer (ViT) model.

        Args:
            concatenate_layers (bool): If True, concatenates the last 4 hidden layers for feature extraction.
                                       If False, uses only the CLS token from the last hidden layer.

        Returns:
            features (np.ndarray): The extracted features from the dataset.
            labels (np.ndarray): The labels corresponding to the features.
        """
        features = []
        labels = []
        with torch.no_grad():
            for images, batch_labels in self.train_loader:
                images = images.to(self.device)
                outputs = self.model(images, output_hidden_states=True)

                if concatenate_layers:
                    last_hidden_states = outputs.hidden_states[-4:]
                    batch_features = torch.cat([hidden_state[:, 0, :] for hidden_state in last_hidden_states], dim=-1)
                else:
                    batch_features = outputs.last_hidden_state[:, 0, :]

                batch_features = batch_features.cpu().numpy()
                features.append(batch_features)
                labels.append(batch_labels)

        return np.concatenate(features), np.concatenate(labels)

    def train_classifiers(self, features, labels, suffix):
        """
        Trains multiple classifiers using the extracted features and saves them as .pkl files.

        Args:
            features (np.ndarray): The extracted features used for training.
            labels (np.ndarray): The labels corresponding to the features.
            suffix (str): A suffix to add to the filename when saving the trained classifiers.

        Returns:
            classifiers (dict): A dictionary of trained classifiers.
        """
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }

        for name, clf in classifiers.items():
            clf.fit(features, labels.argmax(axis=1))
            joblib.dump(clf, f'{name}_{suffix}_classifier.pkl')

        return classifiers

    def train(self):
        """
        Trains classifiers using features extracted from the ViT model with and without concatenating layers.

        The method first extracts features by concatenating the last 4 hidden layers of the ViT model,
        trains classifiers on these features, and then repeats the process using only the CLS token from
        the last hidden layer.

        Returns:
            classifiers_concat (dict): A dictionary of classifiers trained on concatenated layer features.
            classifiers_cls (dict): A dictionary of classifiers trained on CLS token features.
        """
        print("Extracting features with concatenated layers...")
        features_concat, labels = self.extract_features(concatenate_layers=True)
        classifiers_concat = self.train_classifiers(features_concat, labels, 'concat')

        print("Extracting features with CLS token only...")
        features_cls, _ = self.extract_features(concatenate_layers=False)
        classifiers_cls = self.train_classifiers(features_cls, labels, 'cls')

        print("Training completed. Classifiers saved.")
        return classifiers_concat, classifiers_cls
