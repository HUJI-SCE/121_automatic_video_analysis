import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report


class PipeTester:
    def __init__(self, test_dataset, batch_size=32):
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "google/vit-base-patch16-224"
        self.model = ViTModel.from_pretrained(model_name).to(self.device)

    def extract_features(self, concatenate_layers=True):
        features = []
        labels = []
        with torch.no_grad():
            for images, batch_labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images, output_hidden_states=True)

                if concatenate_layers:
                    last_hidden_states = outputs.hidden_states[-4:]
                    batch_features = torch.cat([hidden_state[:, 0, :] for hidden_state in last_hidden_states], dim=-1)
                else:
                    batch_features = outputs.last_hidden_state[:, 0, :]

                batch_features = batch_features.cpu().numpy()
                features.append(batch_features)
                labels.extend(batch_labels.numpy())

        return np.concatenate(features), np.array(labels)

    def test_classifiers(self, classifiers, features, true_labels):
        results = {}
        for name, clf in classifiers.items():
            predictions = clf.predict(features)
            results[name] = (true_labels, predictions)
        return results

    def load_classifiers(self, suffix):
        classifiers = {}
        for name in ['RandomForest', 'SVM', 'XGBoost', 'knn']:
            filename = f'{name}_{suffix}_classifier.pkl'
            if os.path.exists(filename):
                classifiers[name] = joblib.load(filename)
            else:
                print(f"Warning: Classifier file {filename} not found.")
        return classifiers

    def plot_confusion_matrix(self, true_labels, predictions, classifier_name):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.test_dataset.classes,
                    yticklabels=self.test_dataset.classes)
        plt.title(f'Confusion Matrix - {classifier_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{classifier_name}.png')
        plt.close()

    def print_classification_report(self, true_labels, predictions, classifier_name):
        report = classification_report(true_labels, predictions, target_names=self.test_dataset.classes)
        print(f"Classification Report - {classifier_name}")
        print(report)

    def test(self):
        print("Extracting features with concatenated layers...")
        features_concat, true_labels = self.extract_features(concatenate_layers=True)
        classifiers_concat = self.load_classifiers('concat')
        results_concat = self.test_classifiers(classifiers_concat, features_concat, true_labels)

        print("Extracting features with CLS token only...")
        features_cls, _ = self.extract_features(concatenate_layers=False)
        classifiers_cls = self.load_classifiers('cls')
        results_cls = self.test_classifiers(classifiers_cls, features_cls, true_labels)

        # Plot confusion matrices and print classification reports
        for name, (true_labels, predictions) in results_concat.items():
            self.plot_confusion_matrix(true_labels, predictions, f"{name}_concat")
            self.print_classification_report(true_labels, predictions, f"{name}_concat")

        for name, (true_labels, predictions) in results_cls.items():
            self.plot_confusion_matrix(true_labels, predictions, f"{name}_cls")
            self.print_classification_report(true_labels, predictions, f"{name}_cls")
