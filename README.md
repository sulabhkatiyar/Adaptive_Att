# Image Caption Generation using Adaptive Attention

This is a Pytorch implementation of _Adaptive Attention_ model proposed in _Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning_ published in CVPR 2017 [link](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Knowing_When_to_CVPR_2017_paper.pdf)

#### Note: This is a work in progress. I will upload detailed explanation and results as well.

## Experiments
I have used Flickr8k and Flickr30k datasets for experiments. In the paper, authors train both the Caption Generator LSTM and the Encoder CNN (they use ResNet-152 CNN), i.e., they fine-tune the weights of pre-trained ResNet-152 CNN (which was originally trained for Object Recognition task). However, in most methods proposed in literature, the Encoder CNN has not been fine-tuned. Also, most earlier methods have used VGG-16 CNN. The choice of CNN influences Caption Generation performance, as noted in Katiyar et. al. [link](https://arxiv.org/abs/2102.11506). Hence, I have performed two sets of experiments:

(a) Only Decoder (Caption-Generator) is trained with no fine-tuning of CNN. VGG-16 CNN is used as Encoder. 

(b) CNN is fine-tuned with learning rate of 1e-5 and Caption Generator is trained with learning rate of 4e-4. The authors of paper have trained their model for 80 epochs and started CNN fine-tuning after completion of first 20 epochs. However, due to resource constraints I have trained the model for 20 epochs only and I have trained both CNN and Decoder right from the beginning.

The hyperparameter settings used and other implementation details are as follows. For comparison, I have mentioned the settings used in the original implementation by authors, as well.

| Setting | This implementation | Original Implementation |
|---|---|---|
| Decoder Learning Rate | 4e-4 | 5e-4 |
| Decoder Learning Rate | 1e-5 | 1e-5 |
| Total epochs | 20 | 50 |
| Start Fine-Tuning CNN at | 1 epoch | 20 epochs |
| Batch size | 32 | 80 |
| LSTM hidden units | 512 | 512 |
| Early stopping at | 8 epochs | 6 epochs |
| GPU used | NVIDIA Quadro RTX 4000 | Titan X |
