import ml_collections
import numpy as np
import math

import torch
from torch import nn

from models.hybrid_vit import HybridVit


class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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
        self.decoder_channels = [self.decoder_head_channels // 2**i for i in range(self.up_levels + 2)]
        self.n_skip_channels = self.up_levels - 1 
        self.resnet_width = 64 * config.resnet.width_factor
        self.skip_channels = [self.resnet_width * 2**(i + 1) for i in range(1, self.n_skip_channels)[::-1]] + [self.resnet_width]

        # Encoder layers
        self.transformer = HybridVit(config)
        self.transformer.from_pretrained(weights=np.load(config.pre_trained_path))
        
        # Bridge layers
        self.bridge = ASPP(self.hidden_size, self.decoder_head_channels)
        
        # Decoder layers
        self.decoder_blocks = nn.ModuleList()
        for i in range(self.up_levels):
            if i < self.n_skip_channels:
                self.decoder_blocks.append(
                    ResDecoderBlock(self.decoder_channels[i], self.decoder_channels[i + 1], self.skip_channels[i])
                )
            else:
                self.decoder_blocks.append(ResDecoderBlock(self.decoder_channels[i], self.decoder_channels[i + 1]))

        # Output ASPP
        self.aspp_out = ASPP(self.decoder_channels[-2], self.decoder_channels[-1])
                           
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

        # Output ASPP
        x = self.aspp_out(x)
        
        # Final convolution
        return self.conv_final(x)