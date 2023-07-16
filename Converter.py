# Import des librairies utilitaires
from typing import Optional
import torch

from Generator import Generator
from Utils import image_loading_and_pretreatment, images_saving

Tensor = torch.Tensor


class SuperResolutionConverter:

    def __init__(self, model: str = "SRGAN", basic_block_type: Optional[str] = None,
                 pretrained_models_file_name: Optional[str] = None):

        self.model = model
        self.basic_block_type = basic_block_type

        # Définition du générateur et du discriminateur
        self.generator = Generator(model=self.model, basic_block_type=self.basic_block_type)

        # Load pretrained model
        self.generator.load_state_dict(torch.load(f"saved_models/{pretrained_models_file_name}"))

    def apply(self, image_path, hr_size, transformation="resize"):

        # Récupération et prétraitement de l'image hr
        image_hr, image_lr = image_loading_and_pretreatment(image_path, hr_size, transformation=transformation)
        image_lr = image_lr.view(1, *(image_lr.size()))

        # Génération de l'image haute résolution à partir de l'image basse résolution
        image_sr = self.generator(image_lr)

        # Saving results
        file_name = "./results/" + "test"
        images_saving(image_hr, image_lr, image_sr, file_name)
