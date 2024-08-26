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
    """
    A base dataset class for loading and preprocessing images for training or testing.

    Attributes:
        root_dir (str): The root directory where images are stored.
        transform (callable, optional): A function/transform to apply to the images.
        classes (list, optional): A list of class names. Defaults to ['clean', 'faucet', 'welding', 'bubble_air'].
        class_to_idx (dict): A dictionary mapping class names to class indices.
        samples (list): A list of tuples containing image paths and their corresponding class names.
        onehot_encoder (OneHotEncoder): An encoder to convert class labels to one-hot encoded vectors.
    """

    def __init__(self, root_dir, transform=None, classes=None):
        """
        Initializes the BasePipeDataset with the root directory, transforms, and classes.

        Args:
            root_dir (str): The root directory where images are stored.
            transform (callable, optional): A function/transform to apply to the images.
            classes (list, optional): A list of class names. Defaults to ['clean', 'faucet', 'welding', 'bubble_air'].
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes if classes else ['clean', 'faucet', 'welding', 'bubble_air']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

        self.onehot_encoder = OneHotEncoder()
        labels = [[self.class_to_idx[sample[1]]] for sample in self.samples]
        self.onehot_encoder.fit(labels)

    def _load_samples(self):
        """
        Loads image file paths and their corresponding class names from the root directory.

        Returns:
            samples (list): A list of tuples containing image paths and their corresponding class names.
        """
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img in os.listdir(class_dir):
                if img.endswith((".jpg", ".png")):
                    samples.append((os.path.join(class_dir, img), class_name))
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            length (int): The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            image (torch.Tensor): The transformed image tensor.
            label (torch.Tensor): The one-hot encoded label tensor.
        """
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.onehot_encoder.transform([[self.class_to_idx[class_name]]]).toarray()[0]
        return image, torch.tensor(label, dtype=torch.float32)

class PipeTrainDataset(BasePipeDataset):
    """
    A dataset class for loading and preprocessing images for training with data balancing techniques.

    Inherits from BasePipeDataset and adds functionality for data augmentation and sampling.

    Attributes:
        sampling_method (str): The sampling method to use ('downsample' or 'smote').
        downsample_ratio (float): The ratio for downsampling the majority class.
        aug_transforms (transforms.Compose): Composed transformations for data augmentation.
    """

    def __init__(self, root_dir, transform=None, sampling_method='downsample', downsample_ratio=0.75):
        """
        Initializes the PipeTrainDataset with root directory, transforms, and sampling methods.

        Args:
            root_dir (str): The root directory where images are stored.
            transform (callable, optional): A function/transform to apply to the images.
            sampling_method (str, optional): The sampling method to use ('downsample' or 'smote'). Defaults to 'downsample'.
            downsample_ratio (float, optional): The ratio for downsampling the majority class. Defaults to 0.75.
        """
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
        """
        Balances the dataset using the specified sampling method.

        Returns:
            balanced_samples (list): A list of balanced samples after applying the specified sampling method.
        """
        if self.sampling_method == 'downsample':
            return self._downsample()
        elif self.sampling_method == 'smote':
            return self._smote()
        else:
            raise ValueError("Invalid sampling method. Choose 'downsample' or 'smote'.")

    def _downsample(self):
        """
        Downsamples the majority class to balance the dataset.

        Returns:
            balanced_samples (list): A list of samples after downsampling the majority class.
        """
        class_samples = {cls: [] for cls in self.classes}
        for sample in self.samples:
            class_samples[sample[1]].append(sample)

        max_samples = max(len(samples) for samples in class_samples.values())

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
        """
        Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.

        Returns:
            balanced_samples (list): A list of samples after applying SMOTE.
        """
        X = np.array([np.array(Image.open(sample[0]).convert("RGB").resize((64, 64))).flatten() for sample in self.samples])
        y = np.array([self.class_to_idx[sample[1]] for sample in self.samples])

        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        balanced_samples = []
        for x, y in zip(X_resampled, y_resampled):
            class_name = self.classes[y]
            if len(x) == 64 * 64 * 3:  
                original_sample = next((s for s in self.samples if s[1] == class_name), None)
                if original_sample:
                    balanced_samples.append(original_sample)
            else:  
                synthetic_image = Image.fromarray(x.reshape(64, 64, 3).astype(np.uint8))
                balanced_samples.append((synthetic_image, class_name))

        random.shuffle(balanced_samples)
        return balanced_samples

    def __getitem__(self, idx):
        """
        Retrieves an image (either original or synthetic) and its corresponding label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            image (torch.Tensor): The transformed image tensor with data augmentation.
            label (torch.Tensor): The one-hot encoded label tensor.
        """
        img_path_or_image, class_name = self.samples[idx]
        if isinstance(img_path_or_image, str):  
            image = Image.open(img_path_or_image).convert("RGB")
        else: 
            image = img_path_or_image
        image = self.aug_transforms(image)
        if self.transform:
            image = self.transform(image)
        label = self.onehot_encoder.transform([[self.class_to_idx[class_name]]]).toarray()[0]
        return image, torch.tensor(label, dtype=torch.float32)

class PipeTestDataset(Dataset):
    """
    A dataset class for loading and preprocessing images for testing.

    Attributes:
        root_dir (str): The root directory where images are stored.
        transform (callable, optional): A function/transform to apply to the images.
        classes (list): A list of class names.
        class_to_idx (dict): A dictionary mapping class names to class indices.
        samples (list): A list of tuples containing image paths and their corresponding class names.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initializes the PipeTestDataset with the root directory and transforms.

        Args:
            root_dir (str): The root directory where images are stored.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['clean', 'faucet', 'welding', 'bubble_air']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Loads image file paths and their corresponding class names from the root directory.

        Returns:
            samples (list): A list of tuples containing image paths and their corresponding class names.
        """
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img in os.listdir(class_dir):
                if img.endswith((".jpg", ".png")):
                    samples.append((os.path.join(class_dir, img), class_name))
        return samples

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            length (int): The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            image (torch.Tensor): The transformed image tensor.
            label (int): The label index corresponding to the class of the image.
        """
        img_path, class_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[class_name]
        return image, label
