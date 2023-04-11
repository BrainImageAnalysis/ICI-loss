# ICI-loss
Official implementation of Instance-wise and Center-of-Instance (ICI) segmentation loss based on a paper titled **Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function** [[OpenReview]](https://openreview.net/forum?id=8o83y0_YtE&referrer=%5BAuthor%20Console%5D%28/group?id=MIDL.io/2023/Conference/Authors#your-submissions%29)[[PDF-draft]](https://openreview.net/pdf?id=8o83y0_YtE), which was accepted in MIDL 2023 ([oral presentation](https://openreview.net/group?id=MIDL.io/2023/Conference)). Please cite accordingly.

## Abstract
In this paper, we propose a novel two-component loss for biomedical image segmentation tasks called the Instance-wise and Center-of-Instance (ICI) loss, a loss function that addresses the instance imbalance problem commonly encountered when using pixel-wise loss functions such as the Dice loss. The Instance-wise component improves the detection of small instances or "blobs" in image datasets with both large and small instances. The Center-of-Instance component improves the overall detection accuracy. We compared the ICI loss with two existing losses, the Dice loss and the blob loss, in the task of stroke lesion segmentation using the ATLAS R2.0 challenge dataset from MICCAI 2022. Compared to the other losses, the ICI loss provided a better balanced segmentation, and significantly outperformed the Dice loss with an improvement of 1.7−3.7% and the blob loss by 0.6−5.0% in terms of the Dice similarity coefficient on both validation and test set, suggesting that the ICI loss is a potential solution to the instance imbalance problem.

## Available losses

 - Instance-wise and Center-of-Instance (ICI) loss [[code]](https://github.com/BrainImageAnalysis/ICI-loss/blob/main/losses/ICI_loss.py) (see Appendix A and B for formalism)
 - Dual ICI loss [[code]](https://github.com/BrainImageAnalysis/ICI-loss/blob/main/losses/dICI_loss.py) (see Appendix C for formalism)

## How to use
Please see `example_colab.ipynb`, which was written specifically for Google Colaboratory. Also, folder `example_blobs` contains example blobs that are used for visualization in the paper.

## Requirements
The minimum requirements are as follow.
 - [MONAI Core](https://monai.io/core.html)
 - [PyTorch](https://pytorch.org/)

## Auhors

 - Muhammad Febrian Rachmadi (BIA, RIKEN CBS, Japan & Fasilkom UI, Indonesia)
 - Charissa Poon (BIA, RIKEN CBS, Japan)
 - Henrik Skibbe (BIA, RIKEN CBS, Japan)
