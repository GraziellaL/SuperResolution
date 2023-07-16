from typing import Optional
import torch
from torch import nn
from torch import Tensor


class ResidualBlockSRGAN(nn.Module):
    """
    Block résiduel SRGAN
    """
    def __init__(self):
        super(ResidualBlockSRGAN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.BatchNorm2d(64, 0.8),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.BatchNorm2d(64, 0.8),
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection


class DenseBlockESRGAN(nn.Module):
    """
    ESRGAN Dense Block
    """
    def __init__(self, scaling_param: float = 0.2):
        super(DenseBlockESRGAN, self).__init__()
        self.scaling_param = scaling_param
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.PReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(2*64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.PReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(3*64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.PReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(4*64, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.PReLU()
        )
        self.final_conv_bloc = nn.Sequential(
            nn.Conv2d(5*64, 64, kernel_size=3, stride=1, padding=1)  # k3n64s1
        )

    def forward(self, x: Tensor) -> Tensor:
        block1 = self.block1(x)
        block2 = self.block2(torch.cat((x, block1), 1))
        block3 = self.block3(torch.cat((x, block1, block2), 1))
        block4 = self.block4(torch.cat((x, block1, block2, block3), 1))
        final_conv_bloc = self.final_conv_bloc(torch.cat((x, block1, block2, block3, block4), 1))

        return final_conv_bloc.mul(self.scaling_param) + x


class RRDenseBlockESRGAN(nn.Module):
    """
    ESRGAN Residual in Residual Dense Block (RRDB)
    """
    def __init__(self, scaling_param: float = 0.2):
        super(RRDenseBlockESRGAN, self).__init__()
        self.scaling_param = scaling_param
        self.blockDB1 = DenseBlockESRGAN(scaling_param)
        self.blockDB2 = DenseBlockESRGAN(scaling_param)
        self.blockDB3 = DenseBlockESRGAN(scaling_param)

    def forward(self, x: Tensor) -> Tensor:
        blockDB1 = self.blockDB1(x)
        blockDB2 = self.blockDB2(blockDB1)
        blockDB3 = self.blockDB3(blockDB2)

        return blockDB3.mul(self.scaling_param) + x


class UpsamplingBlock(nn.Module):
    def __init__(self):
        super(UpsamplingBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),  # k3n256s1
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self,
                 model: str = "SRGAN",
                 basic_block_type: Optional[str] = None,
                 nbr_basic_blocks: int = 16):

        super(Generator, self).__init__()
        self.model = model

        initial_block = [nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)]  # k3n64s1
        if model == "SRGAN":
            initial_block += [nn.PReLU()]
        self.initial_block = nn.Sequential(*initial_block)

        if self.model == "SRGAN" or basic_block_type == "ResidualBlock":
            basic_block = [ResidualBlockSRGAN() for _ in range(nbr_basic_blocks)]
            self.basic_blocks = nn.Sequential(*basic_block)
        elif basic_block_type == "DenseBlock":
            basic_block = [DenseBlockESRGAN() for _ in range(nbr_basic_blocks)]
            self.basic_blocks = nn.Sequential(*basic_block)
        elif basic_block_type == "ResidualInResidualDenseBlock":
            basic_block = [RRDenseBlockESRGAN() for _ in range(nbr_basic_blocks)]
            self.basic_blocks = nn.Sequential(*basic_block)

        conv_block = [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)]  # k3n64s1
        if model == "SRGAN":
            conv_block += [nn.BatchNorm2d(64, 0.8)]  # d'où sort le 0.8
        self.conv_block = nn.Sequential(*conv_block)

        upsampling_blocks = [UpsamplingBlock() for _ in range(2)]
        self.upsampling_blocks = nn.Sequential(*upsampling_blocks)

        if model == "ESRGAN":
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4),  # k9n3s1
            nn.Sigmoid()  # --> [0, 1]
        )

    def forward(self, x):
        init_block = self.initial_block(x)
        basic_blocks = self.basic_blocks(init_block)
        conv_block = self.conv_block(basic_blocks)
        if self.model == "SRGAN":
            upsampling_blocks = self.upsampling_blocks(conv_block)
        else:  # case "ESRGAN"
            upsampling_blocks = self.upsampling_blocks(init_block + conv_block)
        output = self.final_layer(upsampling_blocks)
        return output

