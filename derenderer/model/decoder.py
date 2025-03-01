"""Based on the Image Captioning model found in 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git
"""

import torch
import torch.nn as nn
import numpy as np

ATTENTION_DIM = 1048 # A
EMBED_DIM = 768 # Emb
DECODER_DIM = 1048 # D
VOCAB_SIZE = 256 + 256 + 3 # V
ENCODER_DIM = 2048 # This depends on the model
DROPOUT = 0.5

PAD, BOS, EOS = 0, 1, 2


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
    

class Decoder(nn.Module):
    """Decoder to go from an encoded image to a sequence of tokens.
    """

    def __init__(self, **params):
        super(Decoder, self).__init__()

        # Parameters. Batch size = B, num pixels = P
        encoder_dim = params.get("encoder_dim", ENCODER_DIM) # E
        attention_dim = params.get("attention_dim", ATTENTION_DIM) # A
        embed_dim = params.get("embed_dim", EMBED_DIM) # Emb
        decoder_dim = params.get("decoder_dim", DECODER_DIM) # D
        vocab_size = params.get("vocab_size", VOCAB_SIZE) # V
        dropout = params.get("dropout", DROPOUT)

        # Layers:
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

        # Store parameters:
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout)


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, 
        for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # (B, E)
        h = self.init_h(mean_encoder_out)  # (B, D)
        c = self.init_c(mean_encoder_out) # (B, D)
        return h, c
    

    def forward(self, enc, max_length, device="cpu"):
        """For inferencing using only the encoded image. Here, enc is
        an encoded vector of dimension (B, P, E).
        """

        B, P, E = enc.shape
        # Initialize LSTM state:
        h, c = self.init_hidden_state(enc)

        # Initial embeddings:
        labels_start = (BOS * torch.ones(B,)).type(torch.int32)
        embs = self.embedding(labels_start) # (B, Emb)

        T_dec = max_length # This time, max length is specified.
        # Create tensors to hold prediction scores and focal map (focal):
        predictions = torch.zeros(B, T_dec).type(torch.int32).to(device)
        focals = torch.zeros(B, T_dec, P).to(device)
    
        # Initialize incomplete indices:
        inds_inc = torch.arange(B)
        inds = torch.arange(B)

        # Let each prediction end at EOS token prediction.
        for t in range(T_dec):
            att_enc, focal = self.attention(enc[inds_inc], h[inds])
            gate = self.sigmoid(self.f_beta(h[inds])) # (B_t, E)
            att_enc = gate * att_enc # Update image with hidden state
            input_lstm = torch.cat([embs, att_enc], dim=1)
            # Update hidden state, cell state. Batch size is changed here
            h, c = self.decode_step(input_lstm, (h[inds], c[inds]))
            preds = self.fc(h)
            focals[inds_inc, t, :] = focal

            # Get token predictions:
            tokens = torch.argmax(preds, dim=1).type(torch.int32)
            predictions[inds_inc, t] = tokens

            # Find which indices in batch have predicted EOS tokens.
            inds = torch.where(tokens != EOS)[0]
            # Update incomplete indices:
            inds_inc = inds_inc[inds]

            # Terminate if there is no remaining indices:
            if inds_inc.size(0) == 0:
                break

            # Update embeddings:
            embs = self.embedding(tokens[inds])
        
        return predictions, focals


    def forcing(self, enc, labels, lengths, device="cpu"):
        """Forward propagation. Uses teacher forcing with the labels.
        - enc: Encoded vector of dimension (B, P, E)
        - labels: Label tokens of dimension (B, T)
        - lengths: Lengths of size (B,), where each value is t <= T
        To be used for training only. For evaluation, we follow the 
        sub-components of the model, taking the max argument token
        each step.
        """

        B, P, E = enc.shape
        V = self.vocab_size
        
        # Sort input data by decreasing lengths.
        lengths, inds_sort = lengths.sort(descending=True)
        enc = enc[inds_sort]
        labels = labels[inds_sort]

        # Embed the labels:
        embs = self.embedding(labels) # (B, T, Emb)
        # Initialize LSTM state:
        h, c = self.init_hidden_state(enc)
        # Decode lengths are lengths - 1
        dec_lengths = (lengths - 1).tolist()
        T_dec = np.max(dec_lengths)

        # Create tensors to hold prediction scores and focal map (focal):
        predictions = torch.zeros(B, T_dec, V).to(device)
        focals = torch.zeros(B, T_dec, P).to(device)

        for t in range(T_dec):
            B_t = sum([l > t for l in dec_lengths]) # Indices are sorted
            # (B_t, E), (B_t, P)
            att_enc, focal = self.attention(enc[:B_t], h[:B_t])
            gate = self.sigmoid(self.f_beta(h[:B_t])) # (B_t, E)
            att_enc = gate * att_enc # Update image with hidden state
            input_lstm = torch.cat([embs[:B_t, t, :], att_enc], dim=1)
            # Update hidden state, cell state
            h, c = self.decode_step(input_lstm, (h[:B_t], c[:B_t]))
            preds = self.fc(self.dropout(h))
            predictions[:B_t, t, :] = preds
            focals[:B_t, t, :] = focal

        return predictions, focals, inds_sort
    

class CaptionerModel(nn.Module):
    """Complete model that includes the encoder and decoder. To be used
    to convert to onnx. Load in the encoder and decoder models first.
    """

    def __init__(self, encoder, decoder, max_length):
        super(CaptionerModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

    
    def forward(self, X):
        """Given a normalized batch image input of dimensions 
        (batch_size, 3, img_size, img_size), feeds it forward.
        """

        imgs_enc = self.encoder(X) # Outputs img, cls encodings
        output, focals = self.decoder(imgs_enc, self.max_length)
        
        return output.cpu().numpy(), focals.cpu().numpy()


def save_model(model, filepath):

    """Saves the model to the file path.
    """

    torch.save(model.state_dict(), filepath)