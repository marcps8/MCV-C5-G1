import logging
import argparse

from torchvision import transforms
from torchvision.datasets import ImageFolder
from network import Network

IMAGES_PATH = "/export/home/group01/mcv/datasets/C3/MIT_split/"
OUTPUT_PATH = "/export/home/group01/MCV-C5-G1/Week3/"


def find_indexs(query_labels, neighbors_labels):
    best_idx = worst_idx = -1
    best_count = worst_count = 0
    for i in range(len(query_labels)):
        curr_label = query_labels[i]
        neighb_labels = neighbors_labels[i][:5]

        count_wrong = 0
        count_corr = 0

        for l in neighb_labels:
            if l != curr_label:
                count_wrong += 1
            else:
                count_corr += 1

        if worst_count <= count_wrong:
            worst_idx = i
            worst_count = count_wrong

        if best_count <= count_corr:
            best_idx = i
            best_count = count_corr

    return best_idx, worst_idx


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


def retrieve(config):
    train_transfs, test_transfs = get_transforms()

    train_dataset = ImageFolder(IMAGES_PATH + "train", transform=train_transfs)
    test_dataset = ImageFolder(IMAGES_PATH + "test", transform=test_transfs)
    train_labels = [x for _, x in train_dataset.samples]

    net = Network(config, train_dataset, test_dataset, train_labels)
    config["load_path"] = OUTPUT_PATH + f"/models/weights_{config['arch_type']}.pth"

    net.load_model()
    query_meta, neighbors_meta, query_labels, neighbors_labels = net.test(
        load_data=True
    )
    print("EVALUATION:\n", net.evaluate(query_labels, neighbors_labels))

    best_idx, worst_index = find_indexs(query_labels, neighbors_labels)

    print("QUERY META (POS):", query_meta[best_idx])
    print("NEIGHBORS LABELS (POS):", neighbors_labels[best_idx][:5])
    print("NEIGHBORS META (POS):", neighbors_meta[best_idx][:5])

    print("QUERY META (NEG):", query_meta[worst_index])
    print("NEIGHBORS LABELS (NEG):", neighbors_labels[worst_index][:5])
    print("NEIGHBORS META (NEG):", neighbors_meta[worst_index][:5])


def evaluate(config):
    train_transfs, test_transfs = get_transforms()

    train_dataset = ImageFolder(IMAGES_PATH + "train", transform=train_transfs)
    test_dataset = ImageFolder(IMAGES_PATH + "test", transform=test_transfs)
    train_labels = [x for _, x in train_dataset.samples]

    net = Network(config, train_dataset, test_dataset, train_labels)
    save_path = "{}/models/weights_{}.pth".format(
        config["out_path"], config["arch_type"]
    )
    net.train(epochs=config["epochs"], save_path=save_path)
    map1, map5, map = net.evaluate()
    print(f"map@1: {map1}, map@5: {map5}, mAP: {map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--arch-type", type=str, default="siamese", choices=["siamese", "triplet"]
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--process", type=str, default="eval", choices=["eval", "retrieve"]
    )
    args = parser.parse_args()
    config = {
        "out_path": args.out_path,
        "embed_size": args.embed_size,
        "batch_size": args.batch_size,
        "arch_type": args.arch_type,
        "epochs": args.epochs,
    }
    logging.getLogger().setLevel(logging.INFO)

    if args.process == "eval":
        evaluate(config)
    else:
        retrieve(config)
