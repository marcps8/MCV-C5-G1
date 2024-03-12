import logging
import argparse

from torchvision import transforms
from torchvision.datasets import ImageFolder
from network import Network

IMAGES_PATH = "/export/home/group01/mcv/datasets/C3/MIT_split/"
OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week3/"


def get_transforms():
    train_transfs = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.3, hue=0.3),
            transforms.RandomResizedCrop(256, (0.15, 1.0)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transfs = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transfs, test_transfs


def main(config):
    train_transfs, test_transfs = get_transforms()

    train_dataset = ImageFolder(IMAGES_PATH + "train", transform=train_transfs)
    test_dataset = ImageFolder(IMAGES_PATH + "test", transform=test_transfs)
    train_labels = [x for _, x in train_dataset.samples]

    net = Network(config, train_dataset, test_dataset, train_labels)
    save_path = "{}/models/weights_{}.pth".format(config["out_path"], config["arch_type"])
    net.train(epochs=config["epochs"], save_path=save_path)
    map1, map5, map = net.test()
    print(f"map@1: {map1}, map@5: {map5}, mAP: {map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-path",
        type=str,
        default=OUTPUT_PATH
    )
    parser.add_argument(
        "--embed-size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--arch-type",
        type=str,
        default="siamese",
        choices=["siamese", "triplet"]
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    config = {
        "out_path": args.out_path,
        "embed_size": args.embed_size,
        "batch_size": args.batch_size,
        "arch_type": args.arch_type,
        "epochs": args.epochs
    }
    logging.getLogger().setLevel(logging.INFO)
    main(config)
