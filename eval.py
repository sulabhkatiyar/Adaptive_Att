import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import json
import argparse


# Parameters
data_folder = 'path_to_data_files'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'   # base name shared by data files
checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = 'path_to_data_files' + '/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
cudnn.benchmark = True 

captions_dump=True


checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

parser = argparse.ArgumentParser(description = 'Evaluation of IC model')
parser.add_argument('beam_size', type=int,  help = 'Beam size for evaluation')
args = parser.parse_args()


def evaluate(beam_size):
    global captions_dump, data_name
    
    empty_hypo = 0
    empty_hypo_r = 0    
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    references = list()
    hypotheses = list()
    captions_dict=dict()

    image_names = list()

    for i, (image, caps, caplens, allcaps, image_name) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size
        image = image.to(device)  

        encoder_out = encoder(image)  
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)

        encoder_out = encoder_out.view(1, -1, encoder_dim)  
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  
        seqs = k_prev_words  

        top_k_scores = torch.zeros(k, 1).to(device)  

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        global_img = decoder.get_global_image(encoder_out)
        h, c = torch.zeros_like(global_img), torch.zeros_like(global_img)
        encoder_out_small = decoder.image_feat_small(encoder_out)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            x_t = torch.cat([embeddings, global_img], dim = 1)
            g_t = decoder.sigmoid(decoder.w_x(x_t) + decoder.w_h(h))
            h, c = decoder.decode_step(x_t,(h, c))     
            s_t = g_t * decoder.tanh(c)
            h_new = decoder.w_g(h).unsqueeze(-1)
            ones_matrix = torch.ones(k, 1, num_pixels).to(device)
            z_t = decoder.w_h_t(decoder.tanh(decoder.w_v(encoder_out_small) + torch.matmul(h_new, ones_matrix))).squeeze(2)
            z_t_ex = decoder.w_h_t(decoder.tanh(decoder.w_s(s_t) + decoder.w_g(h)))
            alpha_t = decoder.softmax(z_t)
            alpha_t_prime = decoder.softmax(torch.cat([z_t, z_t_ex],dim = 1))
            beta_t = alpha_t_prime[:,-1:]
            one_minus_beta = torch.ones(k, 1).to(device) - beta_t
            context_vector = (encoder_out_small * alpha_t.unsqueeze(2)).sum(dim=1)  
            context_vector_prime = (s_t * beta_t) + (context_vector * one_minus_beta)

            scores = decoder.fc(h +  context_vector_prime)  

            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores  

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True) 
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  

            prev_word_inds = top_k_words / vocab_size  
            next_word_inds = top_k_words % vocab_size  
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1) 
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))


            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  

            if k == 0:
                break
            seqs = seqs[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out_small = encoder_out_small[prev_word_inds[incomplete_inds]]
            global_img = global_img[prev_word_inds[incomplete_inds]]

            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            if step > 50:
                break
            step += 1


        try:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        except:
            seq = []
            empty_hypo += 1

        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        image_names.append(image_name)

        assert len(references) == len(hypotheses) == len(image_names)

    # Print the number of hypotheses which remain empty        
    print('The number of empty hypotheses is {}.\n'.format(empty_hypo))

    captions_dict['references']=references
    captions_dict['hypotheses']=hypotheses  
    captions_dict['image_names'] = image_names              

    if captions_dump==True:
        with open('generated_captions_f8k.json', 'w') as gencap:
            json.dump(captions_dict, gencap)
        save_captions_mscoco_format(word_map_file,references,hypotheses,image_names,str(beam_size)+'_f8ktest')

    bleu4 = corpus_bleu(references, hypotheses)
    bleu3 = corpus_bleu(references, hypotheses, (1.0/3.0,1.0/3.0,1.0/3.0,))
    bleu2 = corpus_bleu(references, hypotheses, (1.0/2.0,1.0/2.0,))
    bleu1 = corpus_bleu(references, hypotheses, (1.0/1.0,))


    print("The BLEU scores for model are {}.\n".format([bleu1,bleu2,bleu3,bleu4]))
    with open('eval_run_logs.txt', 'a') as eval_run:
        eval_run.write("For beam-size {} the BLEU scores for model are {}.\n".format(beam_size, [bleu1,bleu2,bleu3,bleu4]))

    return bleu1,bleu2,bleu3,bleu4


def main():
    beam_size = args.beam_size
    was_fine_tuned=False
    scores=evaluate(args.beam_size)
    print("\nBLEU scores @ beam size of %d is %.4f, %.4f, %.4f, %.4f." % (beam_size, scores[0],scores[1],scores[2],scores[3]))
    with open('eval_run_logs.txt', 'a') as eval_run:
        eval_run.write('The model is trained on {dataname} and {was} fine tuned.\n'
                       'The BLEU scores are {bleu_1}, {bleu_2}, {bleu_3}, {bleu_4}.\n'
                       'The beam_size was {beam}.'
                       'The model was trained for {epochs} epochs.\n\n\n'.format(dataname=data_name,
                                                          was ='was' if was_fine_tuned==True else 'was not', 
                                                          bleu_1=scores[0], bleu_2=scores[1], bleu_3=scores[2],
                                                          bleu_4=scores[3], beam=beam_size, epochs=checkpoint['epoch']))

if __name__ == '__main__':
    main()
