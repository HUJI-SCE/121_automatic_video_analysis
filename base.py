import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import random


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
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.samples = self._oversample()
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.33),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def _oversample(self):
        class_samples = {cls: [] for cls in self.classes}
        for sample in self.samples:
            class_samples[sample[1]].append(sample)

        max_samples = max(len(samples) for samples in class_samples.values())
        oversampled = []
        for class_name, samples in class_samples.items():
            oversampled.extend(samples)
            if len(samples) < max_samples:
                oversampled.extend(random.choices(samples, k=max_samples - len(samples)))

        random.shuffle(oversampled)
        return oversampled

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
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