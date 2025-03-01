"""Lists various image encoders for the decoder (stroke estimation) model.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    resnet101, 
    shufflenet_v2_x2_0,
    efficientnet_b0
)

# Encoder selector:
def initialize_encoder(model_name, **params):
    """Creates an encoder with the given model name. The model name is all
    lowercase.
    """
    
    if model_name == "shufflenet":
        return ShufflenetEncoder(**params)
    elif model_name == "resnet":
        return ResnetEncoder(**params)
    elif model_name == "efficientnet":
        return EfficientnetEncoder(**params)
    # Did not select an encoder
    else:
        return


class ImageEncoder(nn.Module):
    """The image encoder module, consisting of a pre-trained model and a
    linear projection layer.
    """

    def __init__(self, model_name, **params):
        super(ImageEncoder, self).__init__()
        # The image net
        imagenet = initialize_encoder(model_name, **params)
        _, res = imagenet.resolution

        # Define linear projection:
        encoder_dim = params.get("encoder_dim")
        self.imagenet = imagenet
        self.proj = nn.Linear(res, encoder_dim, bias=False)


    def forward(self, imgs):
        """Forward pass of an image stack.
        """

        imgs_enc = self.imagenet(imgs)
        out = self.proj(imgs_enc)
        return out


class EfficientnetEncoder(nn.Module):
    """
    EfficientNet image encoder. Use pre-trained ImageNet weights and apply.
    Does not include the pre-processing layers. Has an encoder size of 1280.
    """

    def __init__(self, encoded_image_size=14, **params):
        super(EfficientnetEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        efficientnet = efficientnet_b0(weights='IMAGENET1K_V1')
        modules = list(efficientnet.children())
        self.efficientnet = nn.Sequential(*modules[:-2])

        enc_size = (encoded_image_size, encoded_image_size)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(enc_size)
        self.fine_tune()


    @property
    def resolution(self):
        """The resolution of the output image of shape (B, C, E, E) following
        the efficientnet pass, prior to adaptive pooling. The pooling ratio
        is encoded_image_size / E, and the encoder dimension is C. Returns E, C.
        """

        # Dummy input:
        X = torch.zeros((1, 3, 224, 224)).type(torch.float32)
        with torch.no_grad():
            y = self.efficientnet(X)
        
        _, C, E, _ = y.shape
        return E, C


    @property
    def trainable_parameters(self):
        count = 0
        layers = list(self.efficientnet.children())
        for c in layers:
            for p in c.parameters():
                if p.requires_grad:
                    count += p.nelement()
        
        return count


    def forward(self, imgs):
        out = self.efficientnet(imgs)
        # (batch_size, 978, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 978)
        out = out.permute(0, 2, 3, 1)

        # Flatten out the center:
        B = out.size(0)
        E = out.size(-1)
        out = out.view(B, -1, E)
        return out
    

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 
        2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.efficientnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        layers = list(self.efficientnet.children())
        count = 0
        for c in layers[-3:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                count += p.nelement()


class ShufflenetEncoder(nn.Module):
    """
    Shufflenet image encoder. Use pre-trained ImageNet weights and apply.
    Does not include the pre-processing layers. Has an encoder size of 976.
    """

    def __init__(self, encoded_image_size=14, **params):
        super(ShufflenetEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        shufflenet = shufflenet_v2_x2_0(weights='IMAGENET1K_V1')
        modules = list(shufflenet.children())
        self.shufflenet = nn.Sequential(*modules[:-2])

        enc_size = (encoded_image_size, encoded_image_size)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(enc_size)
        self.fine_tune()


    @property
    def resolution(self):
        """The resolution of the output image of shape (B, C, E, E) following
        the shufflenet pass, prior to adaptive pooling. The pooling ratio
        is encoded_image_size / E, and the encoder dimension is C. Returns E, C.
        """

        # Dummy input:
        X = torch.zeros((1, 3, 224, 224)).type(torch.float32)
        with torch.no_grad():
            y = self.shufflenet(X)
        
        _, C, E, _ = y.shape
        return E, C


    @property
    def trainable_parameters(self):
        count = 0
        layers = list(self.shufflenet.children())
        for c in layers[-3:]:
            for p in c.parameters():
                if p.requires_grad:
                    count += p.nelement()
        
        return count


    def forward(self, imgs):
        out = self.shufflenet(imgs)
        # (batch_size, 978, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 978)
        out = out.permute(0, 2, 3, 1)

        # Flatten out the center:
        B = out.size(0)
        E = out.size(-1)
        out = out.view(B, -1, E)
        return out
    

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 
        2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.shufflenet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        layers = list(self.shufflenet.children())
        count = 0
        for c in layers:
            for p in c.parameters():
                p.requires_grad = fine_tune
                count += p.nelement()


class ResnetEncoder(nn.Module):
    """
    Shufflenet image encoder. Use pre-trained ImageNet weights and apply.
    Does not include the pre-processing layers. Has an encoder size of 976.
    """

    def __init__(self, encoded_image_size=14, **params):
        super(ResnetEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        shufflenet = resnet101(weights='IMAGENET1K_V1')
        modules = list(shufflenet.children())
        self.shufflenet = nn.Sequential(*modules[:-2])

        enc_size = (encoded_image_size, encoded_image_size)
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(enc_size)
        self.fine_tune()


    @property
    def resolution(self):
        """The resolution of the output image of shape (B, C, E, E) following
        the shufflenet pass, prior to adaptive pooling. The pooling ratio
        is encoded_image_size / E, and the encoder dimension is C. Returns E, C.
        """

        # Dummy input:
        X = torch.zeros((1, 3, 224, 224)).type(torch.float32)
        with torch.no_grad():
            y = self.shufflenet(X)
        
        _, C, E, _ = y.shape
        return E, C
    

    @property
    def trainable_parameters(self):
        count = 0
        layers = list(self.resnet.children())
        for c in layers:
            for p in c.parameters():
                if p.requires_grad:
                    count += p.nelement()
        
        return count


    def forward(self, imgs):
        out = self.shufflenet(imgs)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)

        # Flatten out the center:
        B = out.size(0)
        E = out.size(-1)
        out = out.view(B, -1, E)
        return out
    

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 
        2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        layers = list(self.resnet.children())
        count = 0
        for c in layers[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                count += p.nelement()