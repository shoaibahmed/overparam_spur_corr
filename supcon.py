import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConModel(nn.Module):
    def __init__(self, model, dim_in=2048, feat_dim=128):
        super().__init__()
        self.encoder = model
        self.encoder.fc = nn.Identity()  # Remove the FC layer
        
        # self.projector = nn.Linear(dim_in, feat_dim)
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        x = F.normalize(x, dim=1)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, model, feat_dim=128, num_classes=2):
        super().__init__()
        assert isinstance(model, SupConModel)
        self.encoder = model.encoder
        assert isinstance(self.encoder.fc, nn.Identity)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def eval(self):
        # Override the method to keep the model always in the train mode
        self.encoder.train()
        self.classifier.eval()
        return self
