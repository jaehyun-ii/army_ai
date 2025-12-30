# file: model/ntr_model_paper_aligned.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Sub-modules from original U-Net (DoubleConv, Down, Up) ---
# These are the building blocks and remain the same. The `Up` module is
# generic enough to accept any texture we pass it.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv, with texture injection."""
    def __init__(self, up_in_ch, skip_in_ch, out_ch, texture_channels=3):
        super().__init__()
        # Using ConvTranspose as in the original non-bilinear UNet style
        self.up = nn.ConvTranspose2d(up_in_ch, up_in_ch, kernel_size=2, stride=2)
        # Input channels: upsampled + skip_connection + transformed_texture
        self.conv = DoubleConv(up_in_ch + skip_in_ch + texture_channels, out_ch)

    def forward(self, x1, x2, transformed_texture):
        x1 = self.up(x1)
        # Pad x1 to the size of x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # The texture is already transformed and resized, so we just resize it
        # to the current decoder stage size and inject.
        texture_resized = F.interpolate(transformed_texture, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1, texture_resized], dim=1)
        return self.conv(x)

# --- Paper-Aligned Components ---

class TransformationFeatureExtractor(nn.Module):
    """
    A small U-Net that learns to extract transformation parameters (multiplier and adder)
    from the reference image, as described in the ACTIVE paper's supplement.
    """
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        # A lightweight encoder-decoder structure
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv = DoubleConv(64, 32) # 32 from upsample + 32 from skip
        
        # The final layer outputs 2 * n_classes channels:
        # one set for the multiplier, one for the adder.
        self.outc = nn.Conv2d(32, n_classes * 2, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # Decoder
        x = self.up1(x2)
        # Pad and cat with skip connection
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv(x)
        
        combined_output = self.outc(x)
        
        # Split the output into multiplier and adder
        multiplier, adder = torch.chunk(combined_output, 2, dim=1)
        
        # Apply activations to constrain the ranges
        # Multiplier: Sigmoid scaled to [0, 2] to allow dimming or brightening
        multiplier = torch.sigmoid(multiplier) * 2.0 
        # Adder: Tanh to allow adding or subtracting color values in [-1, 1]
        adder = torch.tanh(adder)
        
        return multiplier, adder


class NTR(nn.Module):
    """
    The Neural Texture Renderer, implemented as described in the ACTIVE paper's supplementary material (Fig 5a).
    """
    def __init__(self, n_channels=3, n_classes=3, texture_channels=3):
        super(NTR, self).__init__()
        
        # 1. The Transformation Feature Extractor (TFE)
        self.tfe = TransformationFeatureExtractor(n_channels, n_classes)

        # 2. The Main Rendering U-Net
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        self.up1 = Up(1024, 512, 512, texture_channels)
        self.up2 = Up(512, 256, 256, texture_channels)
        self.up3 = Up(256, 128, 128, texture_channels)
        self.up4 = Up(128, 64, 64, texture_channels)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, ref_image, exp_texture):
        # --- Transformation Path ---
        # 1. Use the TFE to get the transformation parameters from the ref_image
        tf_multiplier, tf_adder = self.tfe(ref_image)
        
        # 2. Resize the expected texture to the full image size
        exp_texture_full = F.interpolate(
            exp_texture, size=ref_image.shape[2:], mode='bilinear', align_corners=False
        )
        
        # 3. Apply the learned transformation to the texture
        # transformed_texture = (exp_texture * multiplier) + adder
        transformed_texture = (exp_texture_full * tf_multiplier) + tf_adder
        
        # --- Main Rendering Path ---
        # 4. Standard U-Net encoder pass on the reference image
        x1 = self.inc(ref_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bottleneck = self.down4(x4)

        # 5. Decoder pass, injecting the *TRANSFORMED* texture at each stage
        x = self.up1(bottleneck, x4, transformed_texture)
        x = self.up2(x, x3, transformed_texture)
        x = self.up3(x, x2, transformed_texture)
        x = self.up4(x, x1, transformed_texture)
        
        # 6. Final output layer with sigmoid for [0, 1] range
        logits = torch.sigmoid(self.outc(x))
        return logits
