import torch.nn as nn
import torch
from SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.atten = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, encoder_input, mask=None):
        encoder_atten_output,  encoder_self_atten = self.atten(encoder_input, encoder_input, encoder_input, mask)
        encoder_output = self.ffn(encoder_atten_output)
        return encoder_output, encoder_self_atten

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_atten = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.encoder_atten = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    def forward(self, encoder_output, decoder_input, decoder_encoder_mask=None, decoder_self_mask=None): 
        decoder_atten_output,  decoder_self_atten = self.atten(decoder_input, decoder_input, decoder_input, decoder_self_mask)
        decode_encoder_output,  decode_encoder_atten = self.atten(decoder_atten_output, encoder_output, encoder_output, decoder_encoder_mask)
        decoder_output = self.ffn(decode_encoder_output)
        return decoder_output, decoder_self_atten, decode_encoder_atten