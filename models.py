import torch
from torch import nn
import torchvision
import pretrainedmodels
import json
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_choice = 1

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=8):
        super(Encoder, self).__init__()        
        self.enc_image_size = encoded_image_size        
        if encoder_choice==1:
            vgg16 = torchvision.models.vgg16(pretrained = True)
            self.features_nopool = nn.Sequential(*list(vgg16.features.children())[:-1])
            self.features_pool = list(vgg16.features.children())[-1]

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):      
        global encoder_choice        
        if encoder_choice==1:
            x = self.features_nopool(images)
            x_pool = self.features_pool(x)
            return x_pool.permute(0,2,3,1)

            
        out = self.adaptive_pool(out)  
        out = out.permute(0, 2, 3, 1)  
        return out
    

class DecoderWithAttention_choice(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):       
        super(DecoderWithAttention_choice, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.final_embeddings_dim = final_embeddings_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.num_feats = 64
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        
        self.decode_step = nn.LSTMCell(embed_dim + decoder_dim, decoder_dim, bias=True)
        self.get_global_features = nn.Linear(encoder_dim, decoder_dim)  
        self.image_feat_small = nn.Linear(encoder_dim, decoder_dim, bias = False)  
        self.w_v = nn.Linear(decoder_dim, self.num_feats, bias = False)  
        self.w_g = nn.Linear(decoder_dim, self.num_feats, bias = False)  
        self.w_h_t = nn.Linear(self.num_feats, 1, bias = False)  
        self.w_x = nn.Linear(decoder_dim + decoder_dim, decoder_dim, bias = False)  
        self.w_h = nn.Linear(decoder_dim, decoder_dim, bias = False)  
        self.w_s = nn.Linear(decoder_dim, self.num_feats, bias = False)  

        self.tanh = torch.nn.Tanh()        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # self.fc = nn.Linear(decoder_dim + decoder_dim, vocab_size)  
        self.fc = nn.Linear(decoder_dim, vocab_size) 

        self.init_weights()  

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

   
    def get_global_image(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        img = self.get_global_features(mean_encoder_out)  
        return img
        
    def forward(self, encoder_out, encoded_captions, caption_lengths):        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)
        assert(self.num_feats == num_pixels)

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)  
        global_img = self.get_global_image(encoder_out)
        h, c = torch.zeros_like(global_img), torch.zeros_like(global_img)

        encoder_out_small = self.image_feat_small(encoder_out)                  
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions_reverse = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)


        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            x_t = torch.cat([embeddings[:batch_size_t, t, :], global_img[:batch_size_t]], dim = 1)
            g_t = self.sigmoid(self.w_x(x_t) + self.w_h(h[:batch_size_t]))
            h, c = self.decode_step(x_t,(h[:batch_size_t], c[:batch_size_t]))     
            s_t = g_t * self.tanh(c)

            h_new = self.w_g(h).unsqueeze(-1)
            ones_matrix = torch.ones(batch_size_t, 1, num_pixels).to(device)
            z_t = self.w_h_t(self.tanh(self.w_v(encoder_out_small[:batch_size_t]) + torch.matmul(h_new, ones_matrix))).squeeze(2)
            z_t_ex = self.w_h_t(self.tanh(self.w_s(s_t[:batch_size_t]) + self.w_g(h[:batch_size_t])))
            alpha_t = self.softmax(z_t)
            alpha_t_prime = self.softmax(torch.cat([z_t, z_t_ex],dim = 1))
            beta_t = alpha_t_prime[:,-1:]
            one_minus_beta = torch.ones(batch_size_t, 1).to(device) - beta_t
            context_vector = (encoder_out_small[:batch_size_t] * alpha_t.unsqueeze(2)).sum(dim=1)           
            context_vector_prime = (s_t * beta_t) + (context_vector * one_minus_beta)

            # preds = self.fc(self.dropout(torch.cat([h, context_vector_prime], dim =1)))  # (batch_size_t, vocab_size)
            preds = self.fc(self.dropout(h + context_vector_prime))  

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind
