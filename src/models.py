import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetUpscale(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        # Encoder (only 1 or 2 downsamples, since input is small)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)  # 11x18 -> 5x9 (floor division)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)  # 5x9 -> 2x4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*4, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )

        # Decoder: Now upsample aggressively
        self.up2 = nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=4, stride=2, padding=1)  # 2->4, 4->8
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*2, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=4, stride=2, padding=1)  # 4->8, 8->16
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )

        # Now upsample from 16x32 approx to target size 103x241 with multiple upscales
        self.upscale_final = nn.Sequential(
            nn.Upsample(size=(103, 241), mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters, base_filters//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters//2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        up2 = self.up2(b)
        cat2 = torch.cat([up2, e2], dim=1)
        d2 = self.dec2(cat2)

        up1 = self.up1(d2)
        cat1 = torch.cat([up1, e1], dim=1)
        d1 = self.dec1(cat1)

        out = self.upscale_final(d1)

        return out
