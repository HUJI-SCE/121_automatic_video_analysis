# main.py
import argparse
from torchvision import transforms
from base import PipeTestDataset, PipeTrainDataset
from training import PipeTrainer
from testing import PipeTester
from transformers import ViTImageProcessor


def main():
    parser = argparse.ArgumentParser(description="Train or test pipe classifiers")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help='Mode: train or test')
    args = parser.parse_args()

    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    ])

    if args.mode == 'train':
        train_dataset = PipeTrainDataset(root_dir="second_data/", transform=transform)
        trainer = PipeTrainer(train_dataset)
        trainer.train()
    elif args.mode == 'test':
        test_dataset = PipeTestDataset(root_dir="first_data/", transform=transform)
        tester = PipeTester(test_dataset)
        tester.test()


if __name__ == "__main__":
    main()
