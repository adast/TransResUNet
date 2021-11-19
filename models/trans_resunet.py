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
    
    
class ResDecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, skip_dim=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=2, stride=2)
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res_conv = ResConv(input_dim + skip_dim, output_dim, 1, 1)
        
    def forward(self, x, skip_connection=None):
        x = self.upsample(x)
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
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
        if 'pre_trained_path' in config:
            self.transformer.from_pretrained(weights=np.load(config.pre_trained_path))
        else:
            print('pre_trained_path is not specified, use this model with torch.load_state_dict only!')
        
        # Bridge layers
        self.bridge = Conv2dReLU(self.hidden_size, self.decoder_head_channels, kernel_size=3, padding='same')
        
        # Decoder layers
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.up_levels):
            if i < self.n_skip_channels:
                self.decoder_blocks.append(
                    ResDecoderBlock(self.decoder_channels[i], self.decoder_channels[i + 1], self.skip_channels[i])
                )
            else:
                self.decoder_blocks.append(ResDecoderBlock(self.decoder_channels[i], self.decoder_channels[i + 1]))
                           
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
        
        # Residual decoder
        for i, block in enumerate(self.decoder_blocks):
            if i < self.n_skip_channels:
                x = block(x, skip_connections[i])
            else:
                x = block(x)
        
        return self.conv_final(x)