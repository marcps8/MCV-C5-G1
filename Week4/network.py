import torch
import pickle
from tqdm import tqdm
from torch import optim
from torchvision import models
import numpy as np

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, embed_size, text_model):
        super(EmbeddingLayer, self).__init__()
        self.linear = torch.nn.Linear(512, embed_size)
        if text_model == "fasttext":
            self.linear_text = torch.nn.Linear(300, embed_size)
        elif text_model == "bert":
            self.linear_text = torch.nn.Linear(768, embed_size)
        self.activation = torch.nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = x.squeeze(-1).squeeze(-1)
        x = self.activation(x)
        x = self.linear(x)
        return x

    def forward_text(self, x):
        x = x.to(self.device)
        x = self.linear_text(x)
        return x


class Network:
    def __init__(self, config, train_loader=None, text_embeddings=None, val_loader = None) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed = EmbeddingLayer(config["embed_size"], config["text_model"]).to(self.device)
        model_img = models.resnet18(weights=True)
        model_img = torch.nn.Sequential(*list(model_img.children())[:-1], self.embed)
        self.model_img = model_img.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_embeddings = text_embeddings

    def train(self, epochs, save_path=None):
        self.loss = torch.nn.TripletMarginLoss(margin=self.config["margin"], p=self.config["p"], eps=self.config["eps"])
        self.optimizer = optim.Adam(self.embed.parameters(), lr=self.config["lr"])
        self.model_img.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for captions, positives, negatives in tqdm(self.train_loader):
                embedded_captions = torch.stack([self.text_embeddings[caption].to(self.device) for caption in captions])
                positives, negatives = (
                    positives.to(self.device),
                    negatives.to(self.device),
                )

                self.optimizer.zero_grad()
                anchor_outs = self.embed.forward_text(embedded_captions)
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
        self.model = torch.load(self.config["load_path"])

    def retrieve(self, save_path = None):
        self.model_img.eval()
        with torch.no_grad():
            db_features, query_features = [], []
            for img, caption, _ in tqdm(self.val_loader):
                img_out = self.model_img(img.to(self.device))
                text_out = self.text_embeddings[caption]
                db_features.append(img_out.cpu().numpy())
                query_features.append(text_out.cpu().numpy())

        db_features = np.concatenate(db_features).astype('float32')
        query_features = np.concatenate(query_features).astype('float32')
        
        if save_path:
            with open(save_path + f"/db_features_{self.config['text_model']}.pkl", "wb") as f:
                pickle.dump(db_features, f)
            with open(save_path + f"/query_features_{self.config['text_model']}.pkl", "wb") as f:
                pickle.dump(query_features, f)