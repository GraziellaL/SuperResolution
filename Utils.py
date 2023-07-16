import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn.functional as torch_fct
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid


class ImageDataset(Dataset):
    """
    Création du dataset de données et pré traitement des images

    Facteur d'échelle de 4× entre les images à basse et haute résolution.
    les images LR sont obtenues par sous-échantillonnage des images HR (BGR, C = 3) en utilisant un noyau bicubique
    avec un facteur de sous-échantillonnage r = 4 = réduction de 16× du nombre de pixels.
    """
    def __init__(self, dataset_path, hr_size, transformation: str = "crop"):
        hr_height, hr_width = hr_size

        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )
        if transformation == "resize":
            self.hr_transform = transforms.Compose(
                [
                    transforms.Resize((hr_height, hr_height)),
                    transforms.ToTensor()
                ]
            )
        else:  # "crop"
            self.hr_transform = transforms.Compose(
                [
                    transforms.RandomCrop((hr_height, hr_height)),
                    transforms.ToTensor()
                ]
            )
        self.hr_shuffle = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip()
                ]
        )
        self.files = sorted(glob.glob(dataset_path + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        img_hr = self.hr_transform(img)
        img_hr = self.hr_shuffle(img_hr)
        img_lr = self.lr_transform(img_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)


def image_loading_and_pretreatment(image_path, hr_size, transformation="resize"):
    img = Image.open(image_path).convert("RGB")

    hr_height, hr_width = hr_size

    # Transforms for low resolution images and high resolution images
    lr_transform = transforms.Compose(
        [
            transforms.Resize((hr_height // 4, hr_height // 4), interpolation=Image.BICUBIC),
        ]
    )
    if transformation == "resize":
        hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height)),
                transforms.ToTensor()
            ]
        )
    else:
        hr_transform = transforms.Compose(
            [
                transforms.CenterCrop((hr_height, hr_height)),
                transforms.ToTensor()
            ]
        )
    img_hr = hr_transform(img)
    img_lr = lr_transform(img_hr)

    return img_hr, img_lr


def convergence_display(loss_dict: dict, title: str):
    """
    Affichage des courbes d'évolution des différentes loss au cours de l'optimisation
    :param loss_dict:
    :param title:
    :return:
    """

    color = ["red", "blue", "green", "black"]
    i = 0
    for name, loss in loss_dict.items():
        i += 1
        if loss:
            plt.plot(loss, color=color[i], label=name)

    # plt.yscale("log")
    plt.autoscale()
    plt.legend(loc="upper right")
    plt.title(title)
    plt.savefig(title)
    plt.show()


def images_saving(images_hr, images_lr, images_sr, file_name):
    images_lr = torch_fct.interpolate(images_lr, scale_factor=4)
    images_sr = make_grid(images_sr, nrow=1, normalize=True)
    images_lr = make_grid(images_lr, nrow=1, normalize=True)
    images_hr = make_grid(images_hr, nrow=1, normalize=True)
    img_grid = torch.cat((images_hr, images_lr, images_sr), -1)
    save_image(img_grid, f"results/{file_name}.png", normalize=False)
