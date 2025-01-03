import torch
import torch.nn as nn
from transformers import ViTModel
 
class LITAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_style = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_attentions=True)
        self.vit_aesthetics = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', output_attentions=True)
        self.linear = nn.Sequential(
            nn.Linear(768*2, 1),
        )
    def forward(self, x):
        style_feature = self.vit_style(x)
        aesthetic_feature = self.vit_aesthetics(x)
        combined = torch.cat((style_feature.last_hidden_state[:, 0, :], aesthetic_feature.last_hidden_state[:, 0, :]), dim=1)
        x = self.linear(combined)
        return x, style_feature.last_hidden_state[:, 0, :], aesthetic_feature.last_hidden_state[:, 0, :]