"""Converts the stroke estimation model into an ONNX model. Must convert
different sections of each model as a separate ONNX model, as many torch
functions are not ONNX compatible.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.transforms import Resize, InterpolationMode
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from derenderer.model.decoder import Decoder, Attention
import onnxruntime

# Default parameters:
ATTENTION_DIM = 1048 # A
EMBED_DIM = 768 # Emb
DECODER_DIM = 1048 # D
VOCAB_SIZE = 256 + 256 + 3 # V
ENCODER_DIM = 2048 # This depends on the model

class EncoderOnnx(nn.Module):
    """Resnet image encoder that is ONNX compatible. 
    Use pre-trained resnet weights and apply.
    """

    def __init__(self, img_size=224, encoded_image_size=14):
        super(EncoderOnnx, self).__init__()

        self.enc_image_size = encoded_image_size
        # pretrained ImageNet ResNet-101
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Replace adaptive pooling 2d with cv2 resize.
        self.img_size = img_size
        self.encoded_image_size = encoded_image_size
        # enc_size = (encoded_image_size, encoded_image_size)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d(enc_size)

    
    def forward(self, images):
        """
        Forward propagation. Here, images is a tensor of dimensions
        (batch_size, 3, image_size, image_size).
        Returns the encoded images.
        """

        # (batch_size, 2048, image_size/32, image_size/32)
        out = self.resnet(images)
        return out
    

    def postprocess(self, enc):
        """Given the ONNX output, processes for inputting into the decoder
        model. Here, enc has shape (batch_size, encoded_dim, patch_size, 
        patch_size), where patch_size = image_size / 32
        """

        # Expand:
        B, C = enc.shape[0], enc.shape[1]
        E = self.enc_image_size
        # This replaces the average adaptive pooling in our case:
        enc_adp = np.zeros((B, C, E, E), dtype=np.float32)
        enc_adp[:, :, ::2, ::2] = enc
        enc_adp[:, :, 1::2, 1::2] = enc
        enc_adp[:, :, ::2, 1::2] = enc
        enc_adp[:, :, 1::2, ::2] = enc

        # enc_out = np.resize(enc, (B, C, E, E))
        
        # Permute:
        enc_out = np.transpose(enc_adp, (0, 2, 3, 1))
        # Flatten out the center
        enc_out = np.reshape(enc_out, (B, -1, C))
        return enc_out.astype(np.float32)
    

    def load(self, model_path):
        """Loads the encoder model from the .pt file path.
        """

        enc_weights = torch.load(model_path, weights_only=True, 
                                 map_location=torch.device('cpu'))
        self.load_state_dict(enc_weights)


    def save(self, onnx_path):
        """Saves the model as an ONNX model.
        """

        # Set to evaluation mode.
        self.eval()

        # Configure input variable:
        C = 3
        H, W = self.img_size, self.img_size

        # Input:
        C = 3
        H = self.img_size
        W = self.img_size
        X = torch.randn(1, C, H, W, requires_grad=True).type(torch.float32)

        # Configure dynamic axis (batch dimension):
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        # Export the model:
        torch.onnx.export(self, X, onnx_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes=dynamic_axes
                          )
        

class DecoderOnnx(nn.Module):
    """Configures the decoder layers and processing so that it is ONNX
    compatible. Saves the decoder layers into multiple onnx files.
    """

    def __init__(self, **params):
        super(DecoderOnnx, self).__init__()

        # Parameters. Batch size = B, num pixels = P
        encoder_dim = params.get("encoder_dim", ENCODER_DIM) # E
        attention_dim = params.get("attention_dim", ATTENTION_DIM) # A
        embed_dim = params.get("embed_dim", EMBED_DIM) # Emb
        decoder_dim = params.get("decoder_dim", DECODER_DIM) # D
        vocab_size = params.get("vocab_size", VOCAB_SIZE) # V

        # Layers:
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Store parameters:
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

    
    def load(self, model_path):
        """Loads the decoder model. Ensure that the same parameters were used.
        """

        dec_weights = torch.load(model_path, weights_only=True, 
                                 map_location=torch.device('cpu'))
        self.load_state_dict(dec_weights)


    # Various save functions:
    def save_embedding(self, onnx_path):
        """Saves the embedding layer.
        """

        model_embed = DecoderEmbedding(self,
                                       vocab_size=self.vocab_size,
                                       embed_dim=self.embed_dim)
        # model_embed.embedding.load_state_dict(self.embedding.state_dict())
        model_embed.save(onnx_path)


    def save_init(self, onnx_path):
        """Saves the embedding layer.
        """

        model_init = DecoderInit(self,
                                 encoder_dim=self.encoder_dim,
                                 decoder_dim=self.decoder_dim)
        # model_init.init_h.load_state_dict(self.init_h.state_dict())
        # model_init.init_c.load_state_dict(self.init_c.state_dict())
        model_init.save(onnx_path)


    def save_iter(self, onnx_path):
        model_iter = DecoderIter(self,
                                 encoder_dim=self.encoder_dim,
                                 decoder_dim=self.decoder_dim,
                                 patch_dim=14*14,
                                 embed_dim=self.embed_dim)
        model_iter.save(onnx_path)
        return model_iter

    def forward(self, enc):
        """A single forward iteration.
        """

        pass


# Define sub-models for the decoder:
class DecoderEmbedding(nn.Module):
    """Embedding model for the decoder.
    """

    def __init__(self, decoder, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM):
        super(DecoderEmbedding, self).__init__()
        self.embedding = decoder.embedding # nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    
    def forward(self, X):
        """Given the input has shape (batch_size,) with integers up to 
        vocab_size, returns the embedding with shape (batch_size, embed_dim).
        """
        
        y = self.embedding(X)
        return y
    
    
    def save(self, onnx_path):
        """Saves the ONNX model file.
        """

        # Set to evaluation mode.
        self.eval()
        X = torch.randint(self.vocab_size, (1,), 
                          requires_grad=False).type(torch.int32)
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        # Export the model:
        torch.onnx.export(self, X, onnx_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes=dynamic_axes
                          )
        
class DecoderInit(nn.Module):
    """Initial h and c linear layers, to be applied to the mean encoded image.
    """

    def __init__(self, decoder, encoder_dim=ENCODER_DIM, decoder_dim=DECODER_DIM):
        super(DecoderInit, self).__init__()
        self.init_h = decoder.init_h # nn.Linear(encoder_dim, decoder_dim)
        self.init_c = decoder.init_c # nn.Linear(encoder_dim, decoder_dim)

        # Store parameters:
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    
    def forward(self, enc_mean):
        """Given the input has shape (batch_size, encoder_dim),
        returns the initial LSTM vectors of shape (batch_size, decoder_dim).
        """
        
        h = self.init_h(enc_mean)
        c = self.init_c(enc_mean)
        return h, c
    

    def save(self, onnx_path):
        """Saves the model as an ONNX file.
        """

        # Set to evaluation mode.
        self.eval()
        E = self.encoder_dim
        X = torch.randn(1, E, requires_grad=True).type(torch.float32)
        example_outputs = self(X)

        # Configure dynamic axis (batch dimension):
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output_h": {0: "batch_size"},
            "output_c": {0: "batch_size"}
        }
        # Export the model:
        torch.onnx.export(self, X, onnx_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=["input"],
                          output_names=["output_h", "output_c"],
                          dynamic_axes=dynamic_axes,
                          example_outputs=example_outputs
                          )
             

    def onnx_preprocess(self, enc):
        """Given an encoded array (batch_size, patch_size, encoded_size),
        pre-processes into an ORT input.
        """

        enc_mean = np.mean(enc, axis=1).astype(np.float32)
        return enc_mean
    

class DecoderIter(nn.Module):
    """Single iteration of the decoder.
    """

    def __init__(self, decoder,
                 encoder_dim=ENCODER_DIM,
                 decoder_dim=DECODER_DIM,
                 embed_dim=EMBED_DIM,
                 patch_dim=14*14):
        super(DecoderIter, self).__init__()
        self.attention = decoder.attention
        self.sigmoid = nn.Sigmoid()
        self.f_beta = decoder.f_beta
        self.decode_step = decoder.decode_step
        self.fc = decoder.fc

        self.patch_dim = patch_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
    

    def forward(self, enc, embs, h, c):
        """Single iteration of the decoder.""" 
        att_enc, focal = self.attention(enc, h)
        gate = self.sigmoid(self.f_beta(h)) # (B_t, E)
        att_enc = gate * att_enc # Update image with hidden state
        input_lstm = torch.cat([embs, att_enc], dim=1)
        # Update hidden state, cell state. Batch size is changed here
        h0, c0 = self.decode_step(input_lstm, (h, c))
        preds = self.fc(h0)

        return preds, h0, c0


    def save(self, onnx_path):
        """Saves the model as an ONNX file.
        """

        # Set to evaluation mode.
        self.eval()
        P = self.patch_dim
        E = self.encoder_dim
        Emb = self.embed_dim
        D = self.decoder_dim
        enc = torch.randn(1, P, E, requires_grad=True).type(torch.float32)
        embs = torch.randn(1, Emb, requires_grad=True).type(torch.float32)
        h = torch.randn(1, D, requires_grad=True).type(torch.float32)
        c = torch.randn(1, D, requires_grad=True).type(torch.float32)

        example_outputs = self(enc, embs, h, c)

        # Configure dynamic axis (batch dimension):
        dynamic_axes = {
            "input_enc": {0: "batch_size"},
            "input_emb": {0: "batch_size"},
            "input_h": {0: "batch_size"},
            "input_c": {0: "batch_size"},
            "output_pred": {0: "batch_size"},
            "output_h": {0: "batch_size"},
            "output_c": {0: "batch_size"}
        }
        input_names = ["input_enc", "input_emb", "input_h", "input_c"]
        output_names = ["output_pred", "output_h", "output_c"]
        # Export the model:
        torch.onnx.export(self, (enc, embs, h, c), onnx_path,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          example_outputs=example_outputs
                          )
        

def init_onnx_session(onnx_path):
    """Start an ONNX inference, for testing.
    """

    providers = ['CPUExecutionProvider']
    ort = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return ort
        