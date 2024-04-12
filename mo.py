import torch
import torch.nn as nn

import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet


def get_activation(name: str):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'softmax':
        return nn.Softmax(dim=1)
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation function: {name}')

class CBM(nn.Module):

    def __init__(self, backbone: ResNet, num_classes: int, num_concepts: int, dropout: float, extra_dim: int, concept_activation: str):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.dropout_prob = dropout
        self.extra_dim = extra_dim

        self.concept_layer = nn.ModuleList([nn.Linear(self.backbone.fc.in_features, 1) for _ in range(num_concepts)])
        self.concept_activation = get_activation(concept_activation)
        self.class_layer = nn.Linear(num_concepts, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        if self.extra_dim > 0:
            self.extra_dim_layer = nn.Linear(self.backbone.fc.in_features, extra_dim)

    def forward(self, x: torch.Tensor, ret_emb: bool=False):
        x = self.get_embedding(x)
        emb = x
        concepts = [self.concept_activation(concept_layer(x)) for concept_layer in self.concept_layer]
        concepts_for_class = torch.cat(concepts, dim=1)
        
        x = self.class_layer(concepts_for_class)

        if ret_emb:
            return concepts, x, emb

        return concepts, x
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.dropout(x)

        x = self.backbone.layer1(x)
        x = self.dropout(x)
        x = self.backbone.layer2(x)
        x = self.dropout(x)
        x = self.backbone.layer3(x)
        x = self.dropout(x)
        x = self.backbone.layer4(x)
        x = self.dropout(x)

        x = self.backbone.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return x