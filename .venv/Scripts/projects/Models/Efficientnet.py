import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, model_save_path, pretrained=True):
        super(EfficientNetModel, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if pretrained:
            self.model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes).to(device)
            self.model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
        else:
            self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes).to(device)
            self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)