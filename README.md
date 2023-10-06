# ICI-loss

An extended version of this loss is available on: [https://github.com/BrainImageAnalysis/instance-loss.git](https://github.com/BrainImageAnalysis/instance-loss.git)

Official implementation of Instance-wise and Center-of-Instance (ICI) segmentation loss based on a paper titled **Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and Center-of-Instance Segmentation Loss Function** [[OpenReview]](https://openreview.net/forum?id=8o83y0_YtE&referrer=%5BAuthor%20Console%5D%28/group?id=MIDL.io/2023/Conference/Authors#your-submissions%29)[[submitted PDF]](https://openreview.net/pdf?id=8o83y0_YtE)[[arXiv]](https://arxiv.org/abs/2304.06229), which was accepted in MIDL 2023 ([oral presentation](https://openreview.net/group?id=MIDL.io/2023/Conference)). Please cite accordingly (please use the arXiv version for now).


## Abstract
In this paper, we propose a novel two-component loss for biomedical image segmentation tasks called the Instance-wise and Center-of-Instance (ICI) loss, a loss function that addresses the instance imbalance problem commonly encountered when using pixel-wise loss functions such as the Dice loss. The Instance-wise component improves the detection of small instances or "blobs" in image datasets with both large and small instances. The Center-of-Instance component improves the overall detection accuracy. We compared the ICI loss with two existing losses, the Dice loss and the blob loss, in the task of stroke lesion segmentation using the ATLAS R2.0 challenge dataset from MICCAI 2022. Compared to the other losses, the ICI loss provided a better balanced segmentation, and significantly outperformed the Dice loss with an improvement of 1.7−3.7% and the blob loss by 0.6−5.0% in terms of the Dice similarity coefficient on both validation and test set, suggesting that the ICI loss is a potential solution to the instance imbalance problem.

## Available losses

 - Instance-wise and Center-of-Instance (ICI) loss [[code]](https://github.com/BrainImageAnalysis/ICI-loss/blob/main/losses/ICI_loss.py) (see Appendix A and B for formalism)
 - Dual ICI loss [[code]](https://github.com/BrainImageAnalysis/ICI-loss/blob/main/losses/dICI_loss.py) (see Appendix C for formalism)

## How to use
Please see `example_colab.ipynb`, which was written specifically for Google Colaboratory. Also, folder `example_blobs` contains example blobs that are used for visualization in the paper.

## Class Arguments

 - `loss_function_pixel`: Any segmentation loss used to calculate the quality of segmentation in pixel-wise level. Written in the original paper as $L_{global}$ in the formalism.
 - `loss_function_instance`: Any segmentation loss used to calculate the quality of segmentation in instance-wise level. Written in the original paper as $L_{instance}$ in the formalism.
 - `loss_function_center`: Any segmentation loss used to calculate the center-of-instance segmentation loss. Written in the original paper as $L_{center}$ in the formalism.
 - `activation`: Set the non-linear function used for segmentation in the last layer of the model. The valid inputs are `"sigmoid"`, `"softmax"`, or `"none"`.  For a two-class segmentation problem (i.e. background and foreground classes), either `"sigmoid"` or `"softmax"` non-linear functions can be called within the function loss. If there are more than two classes, only `"sigmoid"` non-linear function can be called and each channel must be calculated separately. Furthermore, user can always call any non-linear functions outside the function loss and pass both `outputs` and `labels` tensors as probability values (set `activation = "none"`). Default: `"sigmoid"`
 - `num_out_chn`: Number of channels/classes that `outputs` and `labels` tensors (BNHW[D] where N is number of classes). Default: `1`
 - `object_chn`: Channel of `outputs` and `labels` tensors that this loss function will calculate. Note that each channel must be calculated separately. Default: `1`
 - `spatial_dims`: 3 for 3D data (BNHWD) or 2 for 2D data (BNHW). Default: `3`
 - `reduce_segmentation`: Set as `"mean"` if we want to calculate the average of all instance-wise segmentation losses losses or `"sum"` if we want to calculate the sum of all instance-wise segmentation losses. Default: `"mean"`
 - `instance_wise_reduce`: Reducing the instance-wise segmentation losses for all instances (`"instance"`) or batches (`"data"`). Default: `"instance"`
 - `num_iterations`: Number of iterations of max-pooling to perform connected component analysis (CCA). Bigger instances need more iterations (or 1 big instance might be devided into several instances). Bigger image also tend to use more iterations as well. More iterations will use more computational time. Default: 350
 - `segmentation_threshold`: Segmentation threshold to produce binary predicted segmentation before runnning the CCA. Default: `0.5`
 - `max_cc_out`: Maximum numbers of connected components in the `outputs` tensor. This is useful to cut down the computation time and memory usage in the GPUs. This is extremely useful in the early epochs where there are a lot of false predicted segmentation instances. We found that `max_cc_out = 50` produces good performances and time/memory usage. Default: 50
 - `mul_too_many`: Similar to the 'max_cc_out'. We found that `mul_too_many = 50` produces good performances and time/memory usage. Default: 50
 - `min_instance_size`: We can ignore instances that are too small. Set as `0` as the default (i.e. not in use).
 - `centroid_offset`: Offset value to increase the size of center-of-mass for each instance. For example, `centroid_offset = 1` will increase the size of center-of-mass of instance in 2D from `1 x 1` into `3 x 3`. Default: 3 (i.e. center-of-mass's size is either `7 x 7` in 2D or `7 x 7 x 7` in 3D).
 - `smoother`: Used to avoid division by 0 (numerical purposes). Default: `1e-07`
 - `instance_wise_loss_no_tp`: If `True`, the loss function does not include true positive intersections with other instances from the ground truth image (please see Appendix B of the original paper). Default: True (mainly due to successfully improving the performance in DSC). Default: `True`
 - `rate_instead_number`: The loss function will automatically provides the numbers of both missed and false instances. If `False`, the loss function will provide the exact numbers of missed and false instances (e.g. 1 missed and 6 false). If `True`, the loss function will provide the rate of missed and false instances (e.g. 1 / 7 = 0.1429 for missed instances and 6 / 14 = 0.4286 for false instances). Default: `False`

## Requirements
The minimum requirements are as follow.
 - [MONAI Core](https://monai.io/core.html)
 - [PyTorch](https://pytorch.org/)
  
## Auhors

 - Muhammad Febrian Rachmadi (BIA, RIKEN CBS, Japan & Fasilkom UI, Indonesia)
 - Charissa Poon (BIA, RIKEN CBS, Japan)
 - Henrik Skibbe (BIA, RIKEN CBS, Japan)
