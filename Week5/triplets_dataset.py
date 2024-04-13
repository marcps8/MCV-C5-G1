import os

from PIL import Image
from torch.utils.data import Dataset


class TripletsDataset(Dataset):
    def __init__(self, triplets, root_dir, transform=None):
        """
        Args:
            triplets (list of tuples): List of (anchor_caption, positive_id, negative_id, anchor_labels) tuples.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.triplets = triplets
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_caption, positive_id, negative_id = self.triplets[idx]

        # Load images
        positive_image = self.load_image(positive_id)
        negative_image = self.load_image(negative_id)

        if self.transform:
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_caption, positive_image, negative_image

    def load_image(self, image_id):
        if isinstance(image_id, int):
            image_path = os.path.join(
                self.root_dir, f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
            )
        else:
            image_path = f"/export/home/group01/MCV-C5-G1/Week5/generated_images/{image_id}"

        image = Image.open(image_path).convert("RGB")
        return image


class TripletsDatasetVal(Dataset):
    def __init__(self, triplets, root_dir, transform=None):
        """
        Args:
            triplets (list of tuples): List of (anchor_caption, positive_id, negative_id, anchor_labels) tuples.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.triplets = triplets
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_caption, positive_id, _ = self.triplets[idx]
        positive_image = self.load_image(positive_id)

        if self.transform:
            positive_image = self.transform(positive_image)

        return anchor_caption, positive_image

    def load_image(self, image_id):
        image_path = os.path.join(
            self.root_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
        )
        image = Image.open(image_path).convert("RGB")
        return image
