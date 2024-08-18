import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score, \
    balanced_accuracy_score, precision_recall_curve, f1_score, cohen_kappa_score, matthews_corrcoef, fbeta_score
from sklearn.preprocessing import label_binarize


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
            probabilities = clf.predict_proba(features)
            results[name] = (true_labels, predictions, probabilities)
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

    def calculate_additional_metrics(self, true_labels, predictions, probabilities):
        classes = np.unique(true_labels)
        n_classes = len(classes)

        # Binarize the true labels for multi-class metrics
        true_labels_bin = label_binarize(true_labels, classes=classes)

        # Calculate ROC AUC
        if n_classes == 2:
            roc_auc = roc_auc_score(true_labels, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(true_labels_bin, probabilities, average='macro', multi_class='ovr')

        # Calculate PR AUC
        if n_classes == 2:
            pr_auc = average_precision_score(true_labels, probabilities[:, 1])
        else:
            pr_auc = average_precision_score(true_labels_bin, probabilities, average='macro')

        # Calculate Balanced Accuracy
        balanced_acc = balanced_accuracy_score(true_labels, predictions)

        # Calculate G-mean
        recalls = []
        for cls in classes:
            tp = np.sum((true_labels == cls) & (predictions == cls))
            fn = np.sum((true_labels == cls) & (predictions != cls))
            recalls.append(tp / (tp + fn))
        g_mean = np.prod(recalls) ** (1 / n_classes)

        # Calculate F-beta score (beta=2 to weigh recall higher than precision)
        f_beta = fbeta_score(true_labels, predictions, beta=2, average='weighted')

        # Calculate Cohen's Kappa
        cohen_kappa = cohen_kappa_score(true_labels, predictions)

        # Calculate Matthews Correlation Coefficient
        mcc = matthews_corrcoef(true_labels, predictions)

        return {
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc,
            'Balanced Accuracy': balanced_acc,
            'G-mean': g_mean,
            'F-beta Score (beta=2)': f_beta,
            "Cohen's Kappa": cohen_kappa,
            'Matthews Correlation Coefficient': mcc
        }

    def plot_precision_recall_curve(self, true_labels, probabilities, classifier_name):
        classes = np.unique(true_labels)
        n_classes = len(classes)

        plt.figure(figsize=(10, 8))

        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(true_labels, probabilities[:, 1])
            plt.plot(recall, precision, lw=2, label='Precision-Recall curve')
        else:
            true_labels_bin = label_binarize(true_labels, classes=classes)
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(true_labels_bin[:, i], probabilities[:, i])
                plt.plot(recall, precision, lw=2, label=f'Precision-Recall curve of class {classes[i]}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {classifier_name}')
        plt.legend(loc="lower left")
        plt.savefig(f'precision_recall_curve_{classifier_name}.png')
        plt.close()

    def evaluate_classifier(self, true_labels, predictions, probabilities, classifier_name):
        self.plot_confusion_matrix(true_labels, predictions, classifier_name)
        self.print_classification_report(true_labels, predictions, classifier_name)
        self.plot_precision_recall_curve(true_labels, probabilities, classifier_name)

        additional_metrics = self.calculate_additional_metrics(true_labels, predictions, probabilities)
        print(f"\nAdditional Metrics - {classifier_name}")
        for metric, value in additional_metrics.items():
            print(f"{metric}: {value:.4f}")

    def test(self):
        print("Extracting features with concatenated layers...")
        features_concat, true_labels = self.extract_features(concatenate_layers=True)
        classifiers_concat = self.load_classifiers('concat')
        results_concat = self.test_classifiers(classifiers_concat, features_concat, true_labels)

        print("Extracting features with CLS token only...")
        features_cls, _ = self.extract_features(concatenate_layers=False)
        classifiers_cls = self.load_classifiers('cls')
        results_cls = self.test_classifiers(classifiers_cls, features_cls, true_labels)

        for name, (true_labels, predictions, probabilities) in results_concat.items():
            print(f"\nEvaluating {name} classifier (concatenated layers):")
            self.evaluate_classifier(true_labels, predictions, probabilities, f"{name}_concat")

        for name, (true_labels, predictions, probabilities) in results_cls.items():
            print(f"\nEvaluating {name} classifier (CLS token only):")
            self.evaluate_classifier(true_labels, predictions, probabilities, f"{name}_cls")