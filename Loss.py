import torch
from torch import nn
import torch.nn.functional as torch_fct
from torchvision import transforms
from torchvision.models.vgg import vgg19
import matplotlib.pyplot as plt

from Utils import image_loading_and_pretreatment, images_saving

# Pré-traitement de images pour le réseau vgg avec une normalisation suivant les paramètres suivants :
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class VggContentLoss(nn.Module):
    """
    Fonction de perte de VGG

    Indications pytorch :
    Tous les modèles pré-entraînés Vgg attendent des images d'entrée normalisées de la même manière,
    c'est-à-dire des mini-lots d'images RGB à 3 canaux de forme (3 x H x W), (où H et W devraient être au moins 224).
    Les images doivent être chargées dans une plage de [0, 1] puis normalisées à l'aide de
    mean = [0.485, 0.456, 0.406] et std = [0.229, 0.224, 0.225].
    """
    def __init__(self, model):
        super(VggContentLoss, self).__init__()
        self.model = model

        # Modèle VGG19 pre-entrainé sur la base de données ImageNet
        vgg19_model = vgg19(pretrained=True)
        if self.model == "SRGAN":
            # Caractéristiques haut niveau après activation
            self.feature_5_4 = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
            # Caractéristiques bas niveau après activation
            self.feature_2_2 = nn.Sequential(*list(vgg19_model.features.children())[:8]).eval()
        else:  # self.model == "ERSGAN"
            # Caractéristiques haut niveau avant activation
            self.feature_5_4 = nn.Sequential(*list(vgg19_model.features.children())[:34]).eval()
            # Caractéristiques bas niveau avant activation
            self.feature_2_2 = nn.Sequential(*list(vgg19_model.features.children())[:7]).eval()

        # On fige le modèle
        for model_parameters in self.feature_5_4.parameters():
            model_parameters.requires_grad = False
        for model_parameters in self.feature_2_2.parameters():
            model_parameters.requires_grad = False

        # Mise en forme - Normalisation des données d'entrée du modèle VGG
        self.transform = transforms.Normalize(mean=mean, std=std)

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor, feature_maps: str = "VGG_5_4") -> torch.Tensor:

        # Pré-traitement des images
        sr_tensor = self.transform(sr_tensor)
        hr_tensor = self.transform(hr_tensor)

        if feature_maps == "VGG_5_4":
            sr_feature = self.feature_5_4(sr_tensor)
            hr_feature = self.feature_5_4(hr_tensor)
        elif feature_maps == "VGG_2_2":
            sr_feature = self.feature_2_2(sr_tensor)
            hr_feature = self.feature_2_2(hr_tensor)

        content_loss = torch_fct.mse_loss(sr_feature, hr_feature) * 0.0064  # VGG_Loss - Equation (5) - SRGAN

        return content_loss


def features_vgg(hr_tensor):
    """
    VGG 19 features extractor for vizualisation
    :param hr_tensor:
    :return:
    """
    # Modèle VGG19 pre-entrainé sur la base de données ImageNet
    vgg19_model = vgg19(pretrained=True)
    # Caractéristiques haut niveau après activation
    feature_5_4_after = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
    # Caractéristiques bas niveau après activation
    feature_2_2_after = nn.Sequential(*list(vgg19_model.features.children())[:8]).eval()
    # Caractéristiques haut niveau avant activation
    feature_5_4 = nn.Sequential(*list(vgg19_model.features.children())[:34]).eval()
    # Caractéristiques bas niveau avant activation
    feature_2_2 = nn.Sequential(*list(vgg19_model.features.children())[:7]).eval()

    # On fige le modèle
    for model_parameters in feature_5_4.parameters():
        model_parameters.requires_grad = False
    for model_parameters in feature_2_2.parameters():
        model_parameters.requires_grad = False
    for model_parameters in feature_5_4_after.parameters():
        model_parameters.requires_grad = False
    for model_parameters in feature_2_2_after.parameters():
        model_parameters.requires_grad = False

    # Mise en forme - Normalisation des données d'entrée du modèle VGG
    transform = transforms.Normalize(mean=mean, std=std)

    hr_tensor = transform(hr_tensor)
    hr_5_4_after = feature_5_4_after(hr_tensor)
    hr_2_2_after = feature_2_2_after(hr_tensor)
    hr_5_4 = feature_5_4(hr_tensor)
    hr_2_2 = feature_2_2(hr_tensor)

    return [hr_5_4_after, hr_2_2_after, hr_5_4, hr_2_2]


def features_vizualisation(image_path, hr_size, transformation="resize"):
    """
    Features vizualisation
    :param image_path:
    :param hr_size:
    :param transformation:
    :return:
    """
    # Récupération et prétraitement de l'image hr
    image_hr, image_lr = image_loading_and_pretreatment(image_path, hr_size, transformation=transformation)
    features = features_vgg(image_hr)

    processed = []
    for feature in features:
        feature = feature.squeeze(0)
        # Traitement inverse de la moyenne des données ImageNet
        gray_scale = torch.sum(feature, 0)
        gray_scale = gray_scale / feature.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
    plt.savefig(str('features_vizualization.jpg'), bbox_inches='tight')
