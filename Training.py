# Import des librairies utilitaires
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable


from Generator import Generator
from Discriminator import Discriminator
from Loss import VggContentLoss
from Utils import ImageDataset, convergence_display, images_saving

# Paramètres des modèles - Reprise des constantes des publications
params = {
    "SRGAN":
        {
            "basic_block_type": "ResidualBlock",  # Type de block résiduel du générateur
            "hr_size": (96, 96),  # Taille des images haute résolution utilisées
            "lr": 1e-4,  # Learning rate de l'optimiseur Adam
            "beta1": 0.9  # Paramètre beta1 de l'optimiseur Adam
        },
    "ESRGAN":
        {
            "basic_block_type": "ResidualInResidualDenseBlock",
            "hr_size": (96, 96),  # (128, 128),
            "lr": 2*1e-4,  # 2*1e-4 L1 // 1e-4 ESRGAN
            "beta1": 0.9
        }
}


class TrainingSuperResolutionGAN:

    def __init__(self, model: str = "ESRGAN",
                 hr_data_size=(96, 96),
                 batch_size=16, pretrained_models: Optional[str] = None):

        self.model = model
        self.basic_block_type = params[self.model]["basic_block_type"]
        self.batch_size = batch_size

        # Importation des données
        # "./training_data/DIV2K_train_HR"
        self.dataset = ImageDataset("./DIV2K", hr_size=hr_data_size)
        # Dataloader
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Définition du générateur et du discriminateur
        self.generator = Generator(model=self.model, basic_block_type=self.basic_block_type)
        self.discriminator = Discriminator(hr_data_size[0])

        # print(self.generator)

        # Initialisation des réseaux
        if pretrained_models:
            # Load pretrained models
            self.generator.load_state_dict(torch.load(f"saved_models/{pretrained_models}_generator.pth"))
            self.discriminator.load_state_dict(torch.load(f"saved_models/{pretrained_models}_discriminator.pth"))

        # Optimiseur Adam
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=params[self.model]["lr"],
                                            betas=(params[self.model]["beta1"], 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=params[self.model]["lr"],
                                            betas=(params[self.model]["beta1"], 0.999))

        # Loss
        self.VGG_loss = VggContentLoss(self.model)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.MSE_loss = nn.MSELoss()
        self.L1_loss = nn.L1Loss()
        self.generator_loss = {"content_loss": [], "perceptual_loss": [], "adversarial_loss": []}
        self.gan_loss = {"g_loss": [], "d_loss": []}

        # Classe "vrai" = 1
        self.valid = Variable(torch.ones(batch_size), requires_grad=False)
        # Classe "faux" = 0
        self.fake = Variable(torch.zeros(batch_size), requires_grad=False)

    def apply(self, total_epoch: int = 100, loss_type: str = "MSE"):

        for epoch in range(total_epoch):
            for i, images in enumerate(self.dataloader):

                # Images d'entrée
                images_lr = Variable(images["lr"].type(torch.Tensor))
                images_hr = Variable(images["hr"].type(torch.Tensor))

                #########################################
                # Entraîne le générateur
                #########################################
                self.optimizer_G.zero_grad()

                # Génération de l'image haute résolution à partir de l'image basse résolution
                images_sr = self.generator(images_lr)
                # Prédictions
                predictions_sr = self.discriminator(images_sr)
                predictions_hr = self.discriminator(images_hr).detach()

                # Fonction de perte de contenu
                content_loss = None
                perceptual_loss = None
                if self.model == "SRGAN":
                    if loss_type == "MSE":
                        # MSE_Loss entre l'image générée et l'image réelle - Equation (4) - SRGAN
                        content_loss = self.MSE_loss(images_sr, images_hr.detach())
                    elif loss_type == "VGG":
                        # VGG_Loss entre l'image générée et l'image réelle - Equation (5) - SRGAN
                        content_loss = self.VGG_loss(images_sr, images_hr.detach())
                else:  # self.model == "ESRGAN":
                    content_loss = 1e-2 * self.L1_loss(images_sr, images_hr.detach())
                    if loss_type == "VGG":
                        perceptual_loss = self.VGG_loss(images_sr, images_hr.detach())

                # Fonction de perte adverse
                if self.model == "SRGAN":
                    # VGG_Loss - Equation (6) - SRGAN - Objectif : tromper le discriminateur
                    adversarial_loss = 1e-3 * self.adversarial_loss(predictions_sr[:, 0], self.valid)
                else:  # self.model == "ESRGAN":
                    adversarial_loss = 5*1e-3 * self.adversarial_loss(predictions_sr[:, 0] -
                                                             predictions_hr[:, 0].mean(0, keepdim=True), self.valid)

                # Fonction de perte du générateur
                if self.model == "SRGAN":
                    g_loss = content_loss + adversarial_loss  # Equation (3) - SRGAN
                else:  # self.model == "ESRGAN":
                    if loss_type == "PSNR":
                        g_loss = 1e2 * content_loss  # Equation (3) - ESRGAN : j'imagine qu'il aurait fallu utiliser : adversarial_loss + content_loss
                    else:
                        g_loss = perceptual_loss + adversarial_loss + content_loss  # Equation (3) - ESRGAN

                g_loss.backward()
                self.optimizer_G.step()

                #########################################
                #  Entraîne le discriminateur
                #########################################
                self.optimizer_D.zero_grad()

                # Génération de l'image haute résolution à partir de l'image basse résolution puisque
                # le générateur a été mis à jour
                images_sr = self.generator(images_lr).detach()
                # Prédictions
                predictions_sr = self.discriminator(images_sr)
                predictions_hr = self.discriminator(images_hr)

                # Loss des prédictions sur les images réelles et générées
                if self.model == "SRGAN":
                    # Objectif : identifier une réelle image
                    real_loss = self.adversarial_loss(predictions_hr[:, 0], self.valid)
                    # Objectif : identifier une fausse image
                    fake_loss = self.adversarial_loss(predictions_sr[:, 0], self.fake)
                else:  # self.model == "ESRGAN":
                    real_loss = self.adversarial_loss(predictions_hr[:, 0] - predictions_sr[:, 0].mean(0, keepdim=True),
                                                      self.valid)
                    fake_loss = self.adversarial_loss(predictions_sr[:, 0] - predictions_hr[:, 0].mean(0, keepdim=True),
                                                      self.fake)
                # Fonction de perte du discriminateur
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()

                #########################################
                #  Suivi de l'entrainnement
                #########################################

                # Sauvegarde des valeurs des fonctions de pertes pour affichage
                for name, loss in self.generator_loss.items():
                    if name == "content_loss" and content_loss:
                        self.generator_loss["content_loss"].append(content_loss.item())
                    elif name == "perceptual_loss" and perceptual_loss:
                        self.generator_loss["perceptual_loss"].append(perceptual_loss.item())
                    elif name == "adversarial_loss" and adversarial_loss:
                        self.generator_loss["adversarial_loss"].append(adversarial_loss.item())
                self.gan_loss["g_loss"].append(g_loss.item())
                self.gan_loss["d_loss"].append(d_loss.item())

                print(f"[Epoch {epoch}/{total_epoch}] [Batch {i}/{len(self.dataloader)}] "
                      f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                # Sauvegarde de l'avancée des images générées
                sample_interval = 10
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % sample_interval == 0:
                    file_name = f"{self.model}_{self.basic_block_type}_{batches_done*self.batch_size}"
                    images_saving(images_hr, images_lr, images_sr, file_name)

        # sauvegarde des dernières images
        file_name = f"{self.model}_{self.basic_block_type}_results"
        images_saving(images_hr, images_lr, images_sr, file_name)

        # Sauvegarde des modèles obtenus
        torch.save(self.generator.state_dict(),
                   f"{self.model}_{self.basic_block_type}_{loss_type}_{batches_done*self.batch_size}_generator.pth")
        torch.save(self.discriminator.state_dict(),
                   f"{self.model}_{self.basic_block_type}_{loss_type}_{batches_done*self.batch_size}_discriminator.pth")

        # Affichage de la convergence des fonctions de perte
        convergence_display(self.generator_loss, "Différentes loss du générateur")
        convergence_display(self.gan_loss, "Convergence du générateur et du discriminateur")
