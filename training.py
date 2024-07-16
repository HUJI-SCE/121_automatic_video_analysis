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
    def __init__(self, train_dataset, batch_size=32):
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "google/vit-base-patch16-224"
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name, return_tensors='pt')
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def extract_features(self, concatenate_layers=True):
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
        print("Extracting features with concatenated layers...")
        features_concat, labels = self.extract_features(concatenate_layers=True)
        classifiers_concat = self.train_classifiers(features_concat, labels, 'concat')

        print("Extracting features with CLS token only...")
        features_cls, _ = self.extract_features(concatenate_layers=False)
        classifiers_cls = self.train_classifiers(features_cls, labels, 'cls')

        print("Training completed. Classifiers saved.")
        return classifiers_concat, classifiers_cls