import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 5


def build_model(name: str='resnet50', finetune_mode: str='full_ft', drop_rate: float=0.2):
    name = name.lower()
    if name == 'resnet50':
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, NUM_CLASSES)
        )
    elif name == 'mobilenetv2':
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, NUM_CLASSES)
    elif name == 'efficientnetb0':
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, NUM_CLASSES)
    else:
        raise ValueError('Unknown model name')

    # Finetune modes
    if finetune_mode == 'linear_probe':
        for p in net.parameters():
            p.requires_grad = False
        for p in net.parameters():
            # unfreeze classifier head
            p.requires_grad = p.requires_grad or False
        for p in net.parameters():
            pass
        # make sure head is trainable
        if hasattr(net, 'fc'):
            for p in net.fc.parameters(): p.requires_grad = True
        elif hasattr(net, 'classifier'):
            for p in net.classifier.parameters(): p.requires_grad = True

    elif finetune_mode == 'partial_ft':
        # unfreeze last block/stage
        for p in net.parameters():
            p.requires_grad = False
        if hasattr(net, 'layer4'):
            for p in net.layer4.parameters(): p.requires_grad = True
            for p in net.fc.parameters(): p.requires_grad = True
        elif name == 'mobilenetv2':
            for p in net.features[-3:].parameters(): p.requires_grad = True
            for p in net.classifier.parameters(): p.requires_grad = True
        elif name == 'efficientnetb0':
            for p in net.features[-2:].parameters(): p.requires_grad = True
            for p in net.classifier.parameters(): p.requires_grad = True

    else:  # full_ft
        for p in net.parameters():
            p.requires_grad = True

    return net