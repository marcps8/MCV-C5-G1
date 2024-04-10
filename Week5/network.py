import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision.models import ResNet152_Weights, resnet152
from tqdm import tqdm


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size, text_model):
        super(EmbeddingLayer, self).__init__()
        if text_model == "fasttext":
            self.linear = torch.nn.Linear(300, embed_size)
        elif text_model == "bert":
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
        self.embed = EmbeddingLayer(config["embed_size"], config["text_model"]).to(
            self.device
        )
        self.model_img = resnet152(weights=ResNet152_Weights.DEFAULT).to(self.device)
        self.model_img.fc = nn.Identity()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_embeddings = text_embeddings

    def train(self, epochs, save_path=None):
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

        if save_path:
            torch.save(
                self.model_img.state_dict(),
                save_path,
            )

    def load_model(self):
        self.model = torch.load(self.config["load_path"], map_location=self.device)

    def extract_features(self, mode, save_path=None):
        self.model_img.eval()
        with torch.no_grad():
            db_features, query_captions, query_features = [], [], []
            for imgs, captions, _ in tqdm(self.val_loader):
                for i in range(len(captions)):
                    caption = captions[i]
                    # img = imgs[i]
                    # img_out = self.model_img(torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device))
                    text_out = self.text_embeddings[caption]
                    if mode == "text2img":
                        # db_features.append(img_out.cpu().numpy())
                        query_features.append([text_out.cpu().numpy()])
            for captions, imgs, _ in tqdm(self.train_loader):
                for i in range(len(imgs)):
                    img = imgs[i]
                    img_out = self.model_img(
                        torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device)
                    )
                    if mode == "text2img":
                        db_features.append(img_out.cpu().numpy())

        db_features = np.concatenate(db_features).astype("float32")
        query_features = np.concatenate(query_features).astype("float32")

        if save_path:
            with open(
                save_path + f"/db_features_{mode}_{self.config['text_model']}.pkl", "wb"
            ) as f:
                pickle.dump(db_features, f)
            with open(
                save_path + f"/query_features_{mode}_{self.config['text_model']}.pkl",
                "wb",
            ) as f:
                pickle.dump(query_features, f)
