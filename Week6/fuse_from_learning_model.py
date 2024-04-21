# The `CombinedModel` class defines a neural network model that combines image, text, and audio
# features using adapters and fusion weights for multi-modal classification tasks.
import torch.nn as nn 
import torch
import numpy as np

## change to 512 because inception is 512
class Adapter(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if in_features == out_features:
            self.decodding = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU()

            )
        else:
            self.decodding = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=out_features, out_features=out_features),
                nn.ReLU()
            )
            
            
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(0.5)
        self.maintain_projection = nn.Linear(in_features=out_features, out_features=out_features)
        
    def forward(self, x):
        if self.in_features != self.out_features:
            x = torch.relu(self.dropout(self.linear(x)))
            x = self.maintain_projection(x)
            return torch.relu(x)
            
            
        else:
            return torch.relu(self.maintain_projection(x))
            
        
    



class CombinedModel(nn.Module):
    
    def __init__(self, image_in_features:int, audio_in_features:int, text_in_features: int, out_features: int, num_classes: int, aggregation:str="add"):
        super(CombinedModel, self).__init__()
        
        self.fuse_weights = nn.Linear(in_features=out_features*3, out_features=3) #nn.Parameter(nn.init.kaiming_normal(size=(3), device=self.device), requires_grad=True)
        
        self.classification_head = nn.Linear(in_features=out_features, out_features=num_classes)

        # domain_adapters
        self.audio_adapter = Adapter(in_features=audio_in_features, out_features=out_features)
        self.image_adapter = Adapter(in_features=image_in_features, out_features=out_features)
        self.text_adapter = Adapter(in_features=text_in_features, out_features=out_features)

        self.mse = nn.L1Loss(reduction="mean")
        
        self.aggregation = aggregation
    
    def forward(self, x_image, x_text, x_audio, aggregation="add"):
        torch.autograd.set_detect_anomaly(True)
        #n_parameters = (513*128) + (512*513)
        x_audio = self.audio_adapter(x_audio)
        
        #n_parameters (513*2048)+(512*513)
        x_image = self.image_adapter(x_image)
        
        #n_parameters (513*768)+(512*513)
        x_text = self.text_adapter(x_text)
        
        x_combined = torch.cat((x_image, x_text, x_audio), dim=1)
        
        fuse_prob = torch.softmax(self.fuse_weights(x_combined), dim=1)
        x_audio = x_audio * fuse_prob[:, 0].unsqueeze(1)
        x_image = x_image * fuse_prob[:, 1].unsqueeze(1)
        x_text = x_text * fuse_prob[:, 2].unsqueeze(1)
        
        if aggregation == "mean":
            x = (x_audio + x_image + x_text)/3
        if aggregation == "mul":
            x = (x_audio * x_image * x_text)
        if aggregation == "add":
            x = (x_audio + x_image + x_text)
        if aggregation == "concat":
            x = x_combined
        if aggregation == "tensor_fusion":
            x = np.concatenate([x_audio[:, np.newaxis, np.newaxis, :], 
                                x_text[np.newaxis, :, np.newaxis, :],
                                x_image[np.newaxis, np.newaxis, :, :]], axis=-2)
            
        x = torch.relu(x)

        #n_parameters (513*8)
        x = self.classification_head(x)
            
        return x        

    def extract_loss_from_domains(self):
        at_loss = self.mse(self.audio_adapter.maintain_projection.weight, self.text_adapter.maintain_projection.weight)
        ai_loss = self.mse(self.audio_adapter.maintain_projection.weight, self.image_adapter.maintain_projection.weight)
        ti_loss = self.mse(self.text_adapter.maintain_projection.weight, self.image_adapter.maintain_projection.weight)
        return at_loss + ai_loss + ti_loss      