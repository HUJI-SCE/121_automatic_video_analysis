import argparse
from torchvision import transforms
from base import PipeTestDataset, PipeTrainDataset
from training import PipeTrainer
from testing import PipeTester
from transformers import ViTImageProcessor


def main():
    """
    Main function to handle training or testing of pipe classifiers.

    This function parses command-line arguments to determine whether to train or test the classifiers.
    Depending on the mode, it initializes the appropriate dataset, feature extractor, and pipeline (training or testing),
    and then executes the respective pipeline.

    Command-line Arguments:
        --mode: Specifies whether to run the script in 'train' or 'test' mode.

    Actions:
        - If in 'train' mode, it trains classifiers on the training dataset.
        - If in 'test' mode, it tests classifiers on the testing dataset.
    """
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
        train_dataset = PipeTrainDataset(root_dir="worst_case_train/", transform=transform)
        trainer = PipeTrainer(train_dataset)
        trainer.train()
    elif args.mode == 'test':
        test_dataset = PipeTestDataset(root_dir="worst_case_test/", transform=transform)
        tester = PipeTester(test_dataset)
        tester.test()


if __name__ == "__main__":
    """
    The main entry point of the script. 

    When the script is run directly, this block is executed. It calls the `main()` function
    to start the training or testing process based on the provided command-line arguments.
    """
    main()
