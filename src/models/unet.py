import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiamDiffUNet(nn.Module):

    def __init__(self, encoder_name="resnet50", pretrained=True, in_channels=3, num_classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)

        features1 = self.model.encoder(x1)
        features2 = self.model.encoder(x2)

        features = [features2[i] - features1[i] for i in range(1, len(features1))]
        features.insert(0, features2[0])

        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)

        return masks


def build_model(config):
    return SiamDiffUNet(
        encoder_name=config["model"]["encoder_name"],
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
    )