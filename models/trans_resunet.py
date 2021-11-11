import ml_collections
import numpy as np
import math

import torch
from torch import nn

from models.hybrid_vit import HybridVit


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ResConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder_dim, input_decoder_dim, output_dim):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder_dim),
            nn.ReLU(),
            nn.Conv2d(input_encoder_dim, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder_dim),
            nn.ReLU(),
            nn.Conv2d(input_decoder_dim, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
    
    
class DecoderBlock(nn.Module):
    def __init__(self, input_decoder_dim, output_dim, skip_dim=0):
        super().__init__()
        if skip_dim != 0:
            attn_out_dim = max(input_decoder_dim, skip_dim)
            self.attn = AttentionBlock(
                input_encoder_dim=skip_dim, 
                input_decoder_dim=input_decoder_dim, 
                output_dim=attn_out_dim
            )
            self.upsample = nn.ConvTranspose2d(attn_out_dim, attn_out_dim, kernel_size=2, stride=2)
            self.res_conv = ResConv(attn_out_dim + skip_dim, output_dim, 1, 1)
        else:
            self.upsample = nn.ConvTranspose2d(input_decoder_dim, input_decoder_dim, kernel_size=2, stride=2)
            self.res_conv = ResConv(input_decoder_dim, output_dim, 1, 1)
        
    def forward(self, x, skip_connection=None):
        if skip_connection is not None:
            x = self.attn(skip_connection, x)
            x = self.upsample(x)
            x = torch.cat([x, skip_connection], dim=1)
        else:
            x = self.upsample(x)

        return self.res_conv(x)
    
    
class TransResUNet(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        
        # Params
        self.hidden_size = config.transformer.hidden_size
        self.grid_size = (
            config.image_size[0] // config.transformer.patch_size,
            config.image_size[1] // config.transformer.patch_size,
        )
        self.up_levels = (int)(math.log2(config.transformer.patch_size))
        self.decoder_head_channels = config.decoder.head_channels
        self.decoder_channels = [self.decoder_head_channels // 2**i for i in range(self.up_levels + 1)]
        self.n_skip_channels = self.up_levels - 1 
        self.resnet_width = 64 * config.resnet.width_factor
        self.skip_channels = [self.resnet_width * 2**(i + 1) for i in range(1, self.n_skip_channels)[::-1]] + [self.resnet_width]

        # Encoder layers
        self.transformer = HybridVit(config)
        self.transformer.from_pretrained(weights=np.load(config.pre_trained_path))
        
        # Bridge layers
        self.bridge = Conv2dReLU(self.hidden_size, self.decoder_head_channels, kernel_size=3, padding='same')
        
        # Decoder layers
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.up_levels):
            if i < self.n_skip_channels:
                self.decoder_blocks.append(
                    DecoderBlock(
                        input_decoder_dim=self.decoder_channels[i], 
                        output_dim=self.decoder_channels[i + 1], 
                        skip_dim=self.skip_channels[i]
                    )
                )
            else:
                self.decoder_blocks.append(
                    DecoderBlock(
                        input_decoder_dim=self.decoder_channels[i], 
                        output_dim=self.decoder_channels[i + 1]
                    )
                )
                           
        # Final convolution
        self.conv_final = nn.Conv2d(self.decoder_channels[-1], config.n_classes, kernel_size=1)

    def forward(self, pixel_values):
        # Transformer encoder
        x, _, skip_connections = self.transformer(pixel_values)
        x = x[:, 1:, :]
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(-1, self.hidden_size, self.grid_size[0], self.grid_size[1])

        # Bridge
        x = self.bridge(x)

        # Decoder
        for i, block in enumerate(self.decoder_blocks):
            if i < self.n_skip_channels:
                x = block(x, skip_connections[i])
            else:
                x = block(x)
        
        return self.conv_final(x)