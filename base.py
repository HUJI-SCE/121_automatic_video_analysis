import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import random
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class BasePipeDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes if classes else ['clean', 'faucet', 'welding', 'bubble_air']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

        self.onehot_encoder = OneHotEncoder()
        labels = [[self.class_to_idx[sample[1]]] for sample in self.samples]
        self.onehot_encoder.fit(labels)

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img in os.listdir(class_dir):
                if img.endswith((".jpg", ".png")):
                    samples.append((os.path.join(class_dir, img), class_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.onehot_encoder.transform([[self.class_to_idx[class_name]]]).toarray()[0]
        return image, torch.tensor(label, dtype=torch.float32)

class PipeTrainDataset(BasePipeDataset):
    def __init__(self, root_dir, transform=None, sampling_method='downsample', downsample_ratio=0.75):
        super().__init__(root_dir, transform)
        self.sampling_method = sampling_method
        self.downsample_ratio = downsample_ratio
        self.samples = self._balance_samples()
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.33),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def _balance_samples(self):
        if self.sampling_method == 'downsample':
            return self._downsample()
        elif self.sampling_method == 'smote':
            return self._smote()
        else:
            raise ValueError("Invalid sampling method. Choose 'downsample' or 'smote'.")

    def _downsample(self):
        class_samples = {cls: [] for cls in self.classes}
        for sample in self.samples:
            class_samples[sample[1]].append(sample)

        # Find the size of the largest class
        max_samples = max(len(samples) for samples in class_samples.values())

        # Calculate the target size for downsampling
        target_size = int(max_samples * self.downsample_ratio)

        balanced_samples = []
        for class_name, samples in class_samples.items():
            if len(samples) > target_size:
                balanced_samples.extend(random.sample(samples, target_size))
            else:
                balanced_samples.extend(samples)

        random.shuffle(balanced_samples)
        return balanced_samples

    def _smote(self):
        X = np.array([np.array(Image.open(sample[0]).convert("RGB").resize((64, 64))).flatten() for sample in self.samples])
        y = np.array([self.class_to_idx[sample[1]] for sample in self.samples])

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        balanced_samples = []
        for x, y in zip(X_resampled, y_resampled):
            class_name = self.classes[y]
            if len(x) == 64 * 64 * 3:  # Original sample
                original_sample = next((s for s in self.samples if s[1] == class_name), None)
                if original_sample:
                    balanced_samples.append(original_sample)
            else:  # Synthetic sample
                synthetic_image = Image.fromarray(x.reshape(64, 64, 3).astype(np.uint8))
                balanced_samples.append((synthetic_image, class_name))

        random.shuffle(balanced_samples)
        return balanced_samples

    def __getitem__(self, idx):
        img_path_or_image, class_name = self.samples[idx]
        if isinstance(img_path_or_image, str):  # Original image
            image = Image.open(img_path_or_image).convert("RGB")
        else:  # Synthetic image
            image = img_path_or_image
        image = self.aug_transforms(image)
        if self.transform:
            image = self.transform(image)
        label = self.onehot_encoder.transform([[self.class_to_idx[class_name]]]).toarray()[0]
        return image, torch.tensor(label, dtype=torch.float32)

class PipeTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['clean', 'faucet', 'welding', 'bubble_air']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img in os.listdir(class_dir):
                if img.endswith((".jpg", ".png")):
                    samples.append((os.path.join(class_dir, img), class_name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label