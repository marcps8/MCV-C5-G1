import torch
import pickle
from tqdm import tqdm
from torch import optim
from torchvision import models
import numpy as np
import wandb
import os
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from pytorch_metric_learning import losses
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors



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
        #self.model = torch.load(self.config["load_path"])
        self.model = torch.load(self.config["load_path"], map_location=torch.device('cpu'))

    def extract_features(self, mode, save_path = None):
        self.model_img.eval()
        with torch.no_grad():
            db_features, query_captions, query_features = [], [], []
            for imgs, captions, _ in tqdm(self.val_loader):
                for i in range(len(captions)):
                    caption = captions[i]
                    #img = imgs[i]
                    #img_out = self.model_img(torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device))
                    text_out = self.text_embeddings[caption] 
                    if mode == "text2img":
                        #db_features.append(img_out.cpu().numpy())
                        query_features.append([text_out.cpu().numpy()])
            for captions, imgs, _ in tqdm(self.train_loader):
                for i in range(len(imgs)):
                    img = imgs[i]
                    img_out = self.model_img(torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device))
                    if mode == "text2img":
                        db_features.append(img_out.cpu().numpy())
                        
        db_features = np.concatenate(db_features).astype('float32')
        query_features = np.concatenate(query_features).astype('float32')
        
        if save_path:
            with open(save_path + f"/db_features_{mode}_{self.config['text_model']}.pkl", "wb") as f:
                pickle.dump(db_features, f)
            with open(save_path + f"/query_features_{mode}_{self.config['text_model']}.pkl", "wb") as f:
                pickle.dump(query_features, f)

 

class NetworkImg2Text:
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
        
        
        self.batch_size = self.config["batch_size"]
        
        
    def train_online(self, epochs, save_path=None):
        self.loss = losses.AngularLoss(alpha=40) # torch.nn.TripletMarginLoss(margin=self.config["margin"], p=self.config["p"], eps=self.config["eps"])    
        self.optimizer = optim.Adam(self.embed.parameters(), lr=self.config["lr"])
        self.model_img.train()

        wandb.finish()
        wandb.init(project="C5_W4_Img2Txt_A", config=self.config)
        
        for epoch in (range(epochs)):
            running_loss = 0.0
            for idx, (anchors, positives) in tqdm(enumerate(self.train_loader), desc=f"Epoch: {epoch} Mini Batch Training out of {len(self.train_loader)}"):
                self.optimizer.zero_grad()
                anchors = anchors.to(self.device)

                embeddings_images = self.model_img(anchors)

                text_backboned_embeddings = torch.stack([self.text_embeddings[positive].to(self.device) for positive in positives])
                text_embeddings = self.embed.forward_text(text_backboned_embeddings)
                
                distances_text = torch.cdist(text_embeddings, text_embeddings, p=2)
                
                ## select all the possible negatives but ii index
                min_indexes = torch.argmin(distances_text, dim=1) * anchors.shape[0]
                
                mask = torch.ones_like(distances_text.flatten(), dtype=torch.bool)
                mask[min_indexes] = False   
                

                anchors_embeddings = torch.repeat_interleave(embeddings_images, embeddings_images.shape[0], dim=0 )
                positives_emb = torch.repeat_interleave(text_embeddings, text_embeddings.shape[0], dim=0)
                
                
                repeat_embeddings = text_embeddings.repeat(text_embeddings.shape[0], 1)
                negative_embeddings = torch.roll(repeat_embeddings, shifts=1, dims=0)
                
                try:
                    loss = self.loss(anchors_embeddings, positives_emb, negative_embeddings)

                except Exception as e:
                    print(e)
                    print(min_indexes.shape)
                    print(min_indexes)
                    print(anchors_embeddings.shape)
                    print(positives_emb.shape)
                    print(negative_embeddings.shape)
                    
                    negative_embeddings = negative_embeddings[:-1, :]
                    loss = self.loss(anchors_embeddings, positives_emb, negative_embeddings)
                    
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader)}"
            )
            wandb.log({
                    "training_loss": running_loss / len(self.train_loader), 
                    "epoch": epoch + 1,
                    "learning_rate": self.config['lr']})

        wandb.finish()
        if save_path:
            torch.save(
                self.model_img.state_dict(),
                save_path,
            )


    def train(self, epochs, save_path=None):
        self.loss = torch.nn.TripletMarginLoss(margin=self.config["margin"], p=self.config["p"], eps=self.config["eps"])
        self.optimizer = optim.Adam(self.embed.parameters(), lr=self.config["lr"])
        self.model_img.train()

        wandb.finish()
        wandb.init(project="C5_W4_Img2Txt_A", config=self.config)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for anchors, positives, negatives in tqdm(self.train_loader):
                embedded_positive_captions = torch.stack([self.text_embeddings[positive].to(self.device) for positive in positives])
                embedded_negative_captions = torch.stack([self.text_embeddings[negative].to(self.device) for negative in negatives])
                anchors = anchors.to(self.device)
                    

                self.optimizer.zero_grad()
                anchor_outs = self.model_img(anchors)
                pos_outs = self.embed.forward_text(embedded_positive_captions)
                neg_outs = self.embed.forward_text(embedded_negative_captions)
                loss = self.loss(anchor_outs, pos_outs, neg_outs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.train_loader)}"
            )
            wandb.log({
                    "training_loss": running_loss / len(self.train_loader), 
                    "epoch": epoch + 1,
                    "learning_rate": self.config['lr']})

        wandb.finish()
        if save_path:
            torch.save(
                self.model_img.state_dict(),
                save_path,
            )

    def load_model(self):
        self.model = torch.load(self.config["load_path"], map_location=torch.device('cpu'))

    # TODO
    def retrieve(self, mode, save_path = None):
        self.model_img.eval()
        with torch.no_grad():
            db_features, query_captions, query_features = [], [], []
            for imgs, captions, _ in tqdm(self.val_loader):
                for i in range(len(captions)):
                    caption = captions[i]
                    img = imgs[i]
                    img_out = self.model_img(torch.from_numpy(np.expand_dims(img, axis=0)).to(self.device))
                    
                    query_features.append(img_out.cpu().numpy())
                    query_captions.append(caption)
        if mode == "text2img":
            db_features = np.concatenate(query_captions).astype('float32')
  
        query_features = np.concatenate(query_features).astype('float32')
        
        if save_path:
            if mode == "text2img":
                with open(save_path + f"/db_features_{mode}_{self.config['text_model']}.pkl", "wb") as f:
                    pickle.dump(db_features, f)
            else:
                with open(save_path + f"/query_captions_online_img2txt_{self.config['text_model']}.pkl", "wb") as f:
                    pickle.dump(query_captions, f)

            with open(save_path + f"/query_features_online_{mode}_{self.config['text_model']}.pkl", "wb") as f:
                pickle.dump(query_features, f)

    

    

