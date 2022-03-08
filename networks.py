import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.features_extractor = arch_backbone_model(cfg, cfg.backbone_outfeatures, cfg.img_out_features,
                                                      pretrained=cfg.pretrained)

    def forward(self, x):
        img_features = self.features_extractor(x)
        if self.cfg.normalize_img_features:
            img_features = F.normalize(img_features, p=2)
        return img_features


def arch_backbone_model(cfg, num_in_features, num_out_features, pretrained=True):
    arch_name = cfg.backbone_model
    model = torch.hub.load('pytorch/vision:v0.9.0', arch_name, pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(num_in_features, num_out_features)
    return model
