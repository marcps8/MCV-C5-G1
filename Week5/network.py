import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(768, embed_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = self.linear(x)
        return x


class Network:
    def __init__(
        self, config, train_loader=None, text_embeddings=None, val_loader=None
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed = EmbeddingLayer(config["embed_size"]).to(self.device)
        self.model_img = resnet152(weights=ResNet152_Weights.DEFAULT).to(self.device)
        self.model_img.fc = nn.Identity()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_embeddings = text_embeddings

    def train(self, epochs, save_model_path=None, save_embed_path=None):
        self.loss = torch.nn.TripletMarginLoss(margin=self.config["margin"])

        self.optimizer = optim.Adam(self.embed.parameters(), lr=self.config["lr"])
        self.model_img.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for captions, positives, negatives in tqdm(self.train_loader):
                embedded_captions = torch.stack(
                    [
                        self.text_embeddings[caption].to(self.device)
                        for caption in captions
                    ]
                )
                positives, negatives = (
                    positives.to(self.device),
                    negatives.to(self.device),
                )

                self.optimizer.zero_grad()
                anchor_outs = self.embed(embedded_captions)
                pos_outs = self.model_img(positives)
                neg_outs = self.model_img(negatives)
                loss = self.loss(anchor_outs, pos_outs, neg_outs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader)}"
            )

        if save_model_path and save_embed_path:
            torch.save(
                self.model_img.state_dict(),
                save_model_path,
            )
            torch.save(self.embed.state_dict(), save_embed_path)

    def load_model(self):
        self.embed = EmbeddingLayer(self.config["embed_size"])
        self.embed.load_state_dict(
            torch.load(self.config["embed_path"], map_location=self.device)
        )
        self.model_img = resnet152(weights=ResNet152_Weights.DEFAULT).to(self.device)
        self.model_img.fc = nn.Identity()
        self.model_img.load_state_dict(
            torch.load(self.config["model_path"], map_location=self.device)
        )

    def extract_features(self, save_path_db=None, save_path_query=None):
        self.model_img.eval()

        with torch.no_grad():
            db_features, query_features = [], []
            # Batch (32 x ...)
            for captions, imgs in tqdm(self.val_loader):
                # Images
                imgs_out = self.model_img(imgs)
                np_imgs_out = [img_out.cpu().numpy() for img_out in imgs_out]
                db_features.append(np_imgs_out)
                # Captions
                captions_out = [
                    self.embed(self.text_embeddings[caption]) for caption in captions
                ]
                np_captions_out = [
                    caption_out.cpu().numpy() for caption_out in captions_out
                ]
                query_features.append(np_captions_out)

        db_features = np.concatenate(db_features).astype("float32")
        query_features = np.concatenate(query_features).astype("float32")

        if save_path_db and save_path_query:
            with open(save_path_db, "wb") as f:
                pickle.dump(db_features, f)
            with open(save_path_query, "wb") as f:
                pickle.dump(query_features, f)
