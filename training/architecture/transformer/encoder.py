from typing import Any

from timm.layers import StdConv2dSame  # type: ignore
from timm.models.resnetv2 import ResNetV2
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed  # type: ignore

from homr.transformer.configs import Config

from torch import nn
import torch


def get_encoder(config: Config) -> Any:
    backbone_layers = list(config.backbone_layers)
    min_patch_size = 16
    backbone = ResNetV2(
        num_classes=0,
        global_pool="",
        in_chans=config.channels,
        drop_rate=0.1,
        output_stride=min_patch_size,
        drop_path_rate=0.1,
        layers=backbone_layers,
        preact=True,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )

    def embed_layer(**x: Any) -> Any:
        ps = x.pop("patch_size", min_patch_size)
        if ps % min_patch_size != 0 or ps < min_patch_size:
            raise ValueError(
                f"patch_size needs to be multiple of {min_patch_size} with current backbone configuration"  # noqa: E501
            )
        return HybridEmbed(**x, patch_size=ps // min_patch_size, backbone=backbone)

    encoder = VisionTransformer(
        img_size=(config.max_height, config.max_width),
        patch_size=config.patch_size,
        in_chans=config.channels,
        num_classes=0,
        embed_dim=config.encoder_dim,
        depth=config.encoder_depth,
        num_heads=config.encoder_heads,
        embed_layer=embed_layer,
        global_pool="",
    )
    return encoder


def _get_resnet(config: Config) -> ResNetV2:
    """Return the ResNetV2 backbone architecture."""
    backbone_layers = list(config.backbone_layers)
    min_patch_size = 16
    return ResNetV2(
        num_classes=0,
        global_pool="",
        in_chans=config.channels,
        drop_rate=0.1,
        output_stride=min_patch_size,
        drop_path_rate=0.1,
        layers=backbone_layers,
        preact=True,
        stem_type="same",
        conv_layer=StdConv2dSame,
    )


class TransformerEncoderOnly(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = VisionTransformer(
            img_size=(config.max_height, config.max_width),
            patch_size=config.patch_size,
            in_chans=config.channels,
            num_classes=0,
            embed_dim=config.encoder_dim,
            depth=config.encoder_depth,
            num_heads=config.encoder_heads,
            global_pool="",
        )
        # Remove patch embedding since backbone already does this
        del self.encoder.patch_embed

    def forward(self, x):
        # x should already be patch embeddings from your backbone
        # cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.reshape(x, (1, 312, 640))
        x = torch.torch.transpose(x, 1, 2)
        x = torch.cat([self.encoder.cls_token, x], dim=1)
        x += self.encoder.pos_embed
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x


def get_transformer(config: Config) -> nn.Module:
    return TransformerEncoderOnly(config)


class BackboneWithHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # original backbone
        self.backbone = _get_resnet(config)

        # example pooling (if the other model expects flat features)
        self.proj = nn.Conv2d(
            in_channels=2048,
            out_channels=312,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

    def forward(self, x):
        x = self.backbone(x)  # (B, C, H, W)
        x = self.proj(x)
        return x


def get_backbone(config: Config):
    return BackboneWithHead(config)
