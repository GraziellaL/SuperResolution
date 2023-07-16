# Import des librairies utilitaires
from Training import TrainingSuperResolutionGAN, params
from Converter import SuperResolutionConverter
from Loss import features_vizualisation

# Apprentisssage sur cpu
device = "cpu"
batch_size = 5  # 16
total_epoch = 10  # Nombre d'itérations

# Pour le SRGAN, l'apprentissage se fait en commençant par entrainner le générateur avec une perte de contenu MSE
training_MSE_SRGAN = TrainingSuperResolutionGAN(model="SRGAN", batch_size=batch_size)
training_MSE_SRGAN.apply(total_epoch, loss_type="MSE")

##################################################################################

# Les poids obtenus après l'apprentissage avec la perte MSE sont utilisés pour initialiser le GAN
# que l'on continu d'entrainner avec la loss VGG
# training_VGG_SRGAN = TrainingSuperResolutionGAN(model="SRGAN", batch_size=batch_size,
#                                                 pretrained_models="SRGAN_None_MSE_9950")
# training_VGG_SRGAN.apply(total_epoch, loss_type="VGG")

##################################################################################

# Test du réseau SRGAN obtenu sur une image
# converter = SuperResolutionConverter(model="SRGAN",
#                                      pretrained_models_file_name="SRGAN_None_VGG_9995_generator.pth")
# test = converter.apply(image_path="./test_data/0829.png", hr_size=(224, 224))  # params["SRGAN"]["hr_size"])

##################################################################################

# Pour le ESRGAN, l'apprentissage se fait en commençant par entrainner le générateur avec une perte de L1 - PSNR
# training_PSNR_ESRGAN = TrainingSuperResolutionGAN(model="ESRGAN", batch_size=batch_size)
# training_PSNR_ESRGAN.apply(total_epoch, loss_type="PSNR")

##################################################################################

# Les poids obtenus après l'apprentissage avec la perte L1 sont utilisés pour initialiser le GAN
# que l'on continu d'entrainner avec la loss de l'article
# training_VGG_ESRGAN = TrainingSuperResolutionGAN(model="ESRGAN", batch_size=batch_size,
#                                                  pretrained_models="ESRGAN_ResidualInResidualDenseBlock_PSNR_1995")
# training_VGG_ESRGAN.apply(total_epoch, loss_type="VGG")

##################################################################################

##################################################################################

# Visualisation des features du VGG19
# features_vizualisation(image_path="./test_data/0829.png", hr_size=(224, 224), transformation="resize")

