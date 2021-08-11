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
| CNN Learning Rate | 1e-5 | 1e-5 |
| Total epochs | 20 | 50 |
| Start Fine-Tuning CNN at | 1 epoch | 20 epochs |
| Batch size | 32 | 80 |
| LSTM hidden units | 512 | 512 |
| Early stopping at | 8 epochs | 6 epochs |
| GPU used | NVIDIA Quadro RTX 4000 | NVIDIA Titan X |

## Results

The authors have not released results on experiments with Flickr8k dataset. So I cannot compare my results with original implementation. On Flickr8k the results from my implementation can be summarized as:

| CNN | Fine Tune |Beam | BLEU-1 | BLEU-2 | BLEU-3| BLEU-4| METEOR | CIDEr | SPICE | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|---|
| VGG-16 | No | 3 | 0.624 | 0.440 | 0.303 | 0.205 | 0.199 | 0.524 | 0.144 | 0.457 |
| VGG-16 | No | 5 | 0.619 | 0.438 | 0.304 | 0.208 | 0.194 | 0.508 | 0.140 | 0.453 |
| ResNet-152 | Yes | 3 | 0.664 | 0.481| 0.338 | 0.233 | 0.209 | 0.587 | 0.150| 0.477 |
| ResNet-152 | Yes | 5 | 0.659 | 0.480 | 0.339 | 0.235 | 0.207 | 0.589 | 0.150| 0.476|



### Reproducing the results:
1. Download 'Karpathy Splits' for train, validation and testing from [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).
2. For evaluation, the model already generates BLEU scores. In addition, it saves results and image annotations as needed in MSCOCO evaluation format. So for generation of METEOR, CIDEr, ROUGE-L and SPICE evaluation metrics, the evaluation code can be downloaded from [here](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI).

#### Prerequisites:
1. This code has been tested on python 3.6.9 but should word on all python versions > 3.6.
2. Pytorch v1.5.0
3. CUDA v10.1
4. Torchvision v0.6.0
5. Numpy v.1.15.0
6. pretrainedmodels v0.7.4 (Install from [source](https://github.com/Cadene/pretrained-models.pytorch.git)). (I think all versions will work but I have listed here for the sake of completeness.)


#### Execution:
1. First set the path to Flickr8k/Flickr30k/MSCOCO data folders in create_input_files.py file ('dataname' replaced by f8k/f30k/coco).
2. Create processed dataset by running: 
> python create_input_files.py

3. To train the model:
> python train.py

4. To evaluate: 
> python eval.py beamsize 

(eg.: python train_f8k.py 20)

