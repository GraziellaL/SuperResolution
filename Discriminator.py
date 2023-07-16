from torch import nn


class Discriminator(nn.Module):
    def __init__(self, size):
        super(Discriminator, self).__init__()

        # input shape 3 x size x size
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # k3n64s1
            nn.LeakyReLU(0.2),  # alpha = 0.2
        )

        # 64 x size/2 x size/2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # k3n64s2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # k3n128s1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # 128 x size/4 x size/4
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # k3n128s1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # k3n256s1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # 256 x size/8 x size/8
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # k3n256s2
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # k3n512s1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        # 512 x size/16 x size/16
        self.block8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # k3n512s2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)  # inplace=True)
        )

        self.classifier = nn.Sequential(nn.Flatten(),
                                        # 512 x size/16 x size/16
                                        nn.Linear(512 * int(size / 16) * int(size / 16), 1024),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(1024, 1)
                                        # , nn.Sigmoid()  # fonction sigmoid non nécessaire avec BCEWithLogitsLoss
                                        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block7)
        output = self.classifier(block8)
        return output
