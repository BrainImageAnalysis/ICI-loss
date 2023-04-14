import torch
import nibabel as nib
import numpy as np
from .tools import connected_components, connected_components_with_gradients, get_corrected_indices
from monai.losses import DiceLoss

class ICILoss(torch.nn.modules.loss._Loss):
    """
    Compute Instance-wise and Center-of-Instance (ICI) segmentation loss between two tensors.
    The data `outputs` (BNHW[D] where N is number of classes) is compared with ground truth `labels` (BNHW[D]).

    For a two-class segmentation problem (i.e. background and foreground classes), either sigmoid or softmax
    non-linear function can be called within the function loss (set "activation = 'sigmoid'" or "activation = 'softmax'",
    respectively). If there are more than two classes, only sigmoid non-linear function can be called. However, user 
    can always call any non-linear functions outside the function loss and pass both `outputs` and `labels` tensors
    as probability values (set "activation = 'none'"). Note that each channel must be calculated separately

    Connected component analysis (CCA) can be called within the loss function for both `outputs` and `labels` tensors.
    However, CCA for `labels` tensor can be pre-computed outside the loss function (pass a tensor which contains the
    CCA's output to the `cc_label_batch`). The "cc_label_batch = None" is set to be the default where the CCA for 
    `labels` tensor is computed within the loss function.

    The CCA is adapted from Kornia library `here <https://kornia-tutorials.readthedocs.io/en/latest/connected_components.html>`

    The original paper: Improving Segmentation of Objects with Varying Sizes in Biomedical Images using Instance-wise and 
    Center-of-Instance Segmentation Loss Function<https://openreview.net/pdf?id=8o83y0_YtE>
    """
    
    def __init__(
        self,
        loss_function_pixel, 
        loss_function_instance,
        loss_function_center,
        activation = "sigmoid", 
        num_out_chn = 1,
        object_chn = 1,
        spatial_dims = 3,
        reduce_segmentation = "mean",
        instance_wise_reduce = "instance", # or data
        num_iterations = 350, 
        segmentation_threshold = 0.5,
        max_cc_out = 50,
        mul_too_many = 50,
        min_instance_size = 0,
        centroid_offset = 5,
        smoother = 1e-07,
        instance_wise_loss_no_tp = True,
        rate_instead_number = False,
        weighted_instance_wise = False,
        weighted_fdr = False,
    ):
        """
        Args:
            loss_function_pixel: Any segmentation loss used to calculate the quality of segmentation in pixel-wise level.
                Written in the original paper as L_{global} in the formalism. 
            loss_function_instance: Any segmentation loss used to calculate the quality of segmentation in instance-wise level.
                Written in the original paper as L_{instance} in the formalism.
            loss_function_center: Any segmentation loss used to calculate the center-of-instance segmentation loss.
                Written in the original paper as L_{center} in the formalism.
            activation: Set the non-linear function used for segmentation in the last layer of the model.
                The valid inputs are "sigmoid", "softmax", or "none". Default: "sigmoid"
            num_out_chn: Number of channels/classes that `outputs` and `labels` tensors (BNHW[D] where N is number of classes).
                Default: 1
            object_chn: Channel of `outputs` and `labels` tensors that this loss function will calculate. Note that each channel
                must be calculated separately. Default: 1
            spatial_dims: 3 for 3D data (BNHWD) or 2 for 2D data (BNHW). Default: 3
            reduce_segmentation: Set as "mean" if we want to calculate the average of all instance-wise segmentation losses 
                losses or "sum" if we want to calculate the sum of all instance-wise segmentation losses. Default: "mean"
            instance_wise_reduce: Reducing the instance-wise segmentation losses for all instances ("instance") or batches ("data"). 
                Default: "instance"
            num_iterations: Number of iterations of max-pooling to perform connected component analysis (CCA). Bigger instances
                need more iterations (or 1 big instance might be devided into several instances). Bigger image also tend to use
                more iterations as well. More iterations will use more computational time. Default: 350
            segmentation_threshold: Segmentation threshold to produce binary predicted segmentation before runnning the CCA.
                Default: 0.5
            max_cc_out: Maximum numbers of connected components in the `outputs` tensor. This is useful to cut down the computation
                time and memory usage in the GPUs. This is extremely useful in the early epochs where there are a lot of false
                predicted segmentation instances. We found that `max_cc_out = 50` produces good performances and time/memory usage.
                Default: 50
            mul_too_many: Similar to the 'max_cc_out'.  We found that `mul_too_many = 50` produces good performances and time/memory usage.
                 Default: 50
            min_instance_size: We can ignore instances that are too small. Set as 0 as the default (i.e. not in use).
            centroid_offset: Offset value to increase the size of center-of-mass for each instance. For example, `centroid_offset = 1` will
                increase the size of center-of-mass of instance in 2D from `1 x 1` into `3 x 3`. Default: 3 (i.e. center-of-mass's size is
                either `7 x 7` in 2D or `7 x 7 x 7` in 3D).
            smoother: Used to avoid division by 0 (numerical purposes).
            instance_wise_loss_no_tp: If `True`, the loss function does not include true positive intersections with other instances 
                from the ground truth image (please see Appendix B of the original paper). Default: True (mainly due to successfully 
                improving the performance in DSC). Default: True
            rate_instead_number: The loss function will automatically provides the numbers of both missed and false instances.
                If `False`, the loss function will provide the exact numbers of missed and false instances (e.g. 1 missed and 6 false).
                If `True`, the loss function will provide the rate of missed and false instances (e.g. 1 / 7 = 0.1429 for missed instances
                and 6 / 14 = 0.4286 for false instances).
            weighted_instance_wise: Not currently used and never been tested, so please do not use it. Default to False.
            weighted_fdr: Not currently used and never been tested, so please do not use it. Default to False.

        Raises:
            ValueError: When ``activation`` is not one of ["sigmoid", "softmax", "none"].
            ValueError: When ``reduce_segmentation`` is not one of ["mean", "sum"].
            ValueError: When ``instance_wise_reduce`` is not one of ["instance", "data"].

        """
        super().__init__()
        self.loss_function_pixel = loss_function_pixel
        self.loss_function_instance = loss_function_instance
        self.loss_function_center = loss_function_center
        self.num_out_chn = num_out_chn
        self.object_chn = object_chn
        self.spatial_dims = spatial_dims
        self.activation = activation
        self.reduce = reduce_segmentation
        self.instance_wise_reduce = instance_wise_reduce
        self.num_iterations = num_iterations 
        self.max_cc_out = max_cc_out
        self.min_instance_size = min_instance_size
        self.segmentation_threshold = segmentation_threshold
        self.centroid_offset = centroid_offset
        self.mul_too_many = mul_too_many
        self.smoother = smoother
        self.instance_wise_loss_no_tp = instance_wise_loss_no_tp
        self.weighted_instance_wise = weighted_instance_wise
        self.weighted_fdr = weighted_fdr
        self.rate_instead_number = rate_instead_number

        if self.activation != "sigmoid" and self.activation != "softmax" and self.activation != "none":
            raise ValueError("The valid inputs for `activation` are 'sigmoid', 'softmax', or 'none'.")

        if self.reduce != "mean" and self.reduce != "sum":
            raise ValueError("The valid inputs for `reduce_segmentation` are 'mean' or 'sum'.")

        if self.instance_wise_reduce != "instance" and self.instance_wise_reduce != "data":
            raise ValueError("The valid inputs for `instance_wise_reduce` are 'instance' or 'data'.")
        
    def print_parameters(self):
        """
        Print all parameters currently being used by the loss function.
        """
        print(
            f"\nVALUES OF PARAMETERS"
            f"\n -> loss_function_pixel: {self.loss_function_pixel}"
            f"\n -> loss_function_instance: {self.loss_function_instance}"
            f"\n -> loss_function_center: {self.loss_function_center}"
            f"\n -> num_out_chn: {self.num_out_chn}"
            f"\n -> object_chn: {self.object_chn}"
            f"\n -> spatial_dims: {self.spatial_dims}"
            f"\n -> activation: {self.activation}"
            f"\n -> reduce: {self.reduce}"
            f"\n -> instance_wise_reduce: {self.instance_wise_reduce}"
            f"\n -> num_iterations: {self.num_iterations}"
            f"\n -> max_cc_out: {self.max_cc_out}"
            f"\n -> min_instance_size: {self.min_instance_size}"
            f"\n -> segmentation_threshold: {self.segmentation_threshold}"
            f"\n -> centroid_offset: {self.centroid_offset}"
            f"\n -> mul_too_many: {self.mul_too_many}"
            f"\n -> smoother: {self.smoother}"
            f"\n -> instance_wise_loss_no_tp: {self.instance_wise_loss_no_tp}"
            f"\n -> weighted_instance_wise: {self.weighted_instance_wise}"
            f"\n -> weighted_fdr: {self.weighted_fdr}"
            f"\n -> rate_instead_number: {self.rate_instead_number}"
        )
        
    def edit_parameters(
        self,
        loss_function_pixel = None,
        loss_function_instance = None,
        loss_function_center = None,
        num_out_chn = None,
        object_chn = None,
        spatial_dims = None,
        activation = None, 
        reduce_segmentation = None, # or sum
        instance_wise_reduce = None, # or data
        num_iterations = None, 
        max_cc_out = None,
        min_instance_size = None,
        segmentation_threshold = None,
        centroid_offset = None,
        mul_too_many = None,
        smoother = None,
        instance_wise_loss_no_tp = None,
        rate_instead_number = None,
        weighted_instance_wise = None,
        weighted_fdr = None,
    ):
        """
        All parameters can be edited/updated on the fly, so users do not have to create different instance of the loss function.
        Arguments are the same as in the initialization, but the user can choose any parameter that will be edited/updated. 
        """
        if loss_function_pixel is not None:
            self.loss_function_pixel = loss_function_pixel
            
        if loss_function_instance is not None:
            self.loss_function_instance = loss_function_instance
            
        if loss_function_center is not None:
            self.loss_function_center = loss_function_center
            
        if num_out_chn is not None:
            self.num_out_chn = num_out_chn
            
        if object_chn is not None:
            self.object_chn = object_chn
            
        if spatial_dims is not None:
            self.spatial_dims = spatial_dims
            
        if activation is not None:
            self.activation = activation
            
        if reduce_segmentation is not None:
            self.reduce = reduce_segmentation
            
        if instance_wise_reduce is not None:
            self.instance_wise_reduce = instance_wise_reduce
            
        if num_iterations is not None:
            self.num_iterations = num_iterations 
            
        if max_cc_out is not None:
            self.max_cc_out = max_cc_out
            
        if min_instance_size is not None:
            self.min_instance_size = min_instance_size
            
        if segmentation_threshold is not None:
            self.segmentation_threshold = segmentation_threshold
            
        if centroid_offset is not None:
            self.centroid_offset = centroid_offset
                        
        if mul_too_many is not None:
            self.mul_too_many = mul_too_many
            
        if smoother is not None:
            self.smoother = smoother

        if instance_wise_loss_no_tp is not None:
            self.instance_wise_loss_no_tp = instance_wise_loss_no_tp
            
        if weighted_instance_wise is not None:
            self.weighted_instance_wise = weighted_instance_wise
            
        if weighted_fdr is not None:
            self.weighted_fdr = weighted_fdr

        if rate_instead_number is not None:
            self.rate_instead_number = rate_instead_number
                
    def forward(self, outputs, labels, cc_label_batch=None):
        """
        Args:
            outputs: The predicted segmentation. The shape should be BNH[WD], where N is the number of classes.
            labels: The manual label/mask. The shape should be BNH[WD] or B1H[WD], where N is the number of classes.
            cc_label_batch: The pre-computed CCA of the `labels`. If `cc_label_batch = None` the loss function
                will perform CCA to the `labels` tensor.

            WARNING: The loss function does not check whether the shapes of `outputs`, `labels`, and `cc_label_batch`
                are the same.

        Outputs:
            1. Pixel-wise segmentation loss
            2. Instance-wise segmentation loss
            3. Center-of-instance segmentation loss
            4. Pixel-wise false detection rate (FDR)
            5. Number/rate of false instances
            6. Number/rate of missed instances

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from ICILoss.ICI_loss import ICILoss
            >>> from monai.losses import DiceLoss
            >>> from monai.data.synthetic import create_test_image_3d
            >>> loss_dice = DiceLoss(
            >>>         to_onehot_y=False, 
            >>>         sigmoid=False, 
            >>>         softmax=False
            >>>     )
            >>> _, lbl = create_test_image_3d(128, 128, 128, num_seg_classes=1)
            >>> _, out = create_test_image_3d(128, 128, 128, num_seg_classes=1)
            >>> lbl = torch.reshape(torch.tensor(lbl).float(), (1, 1, 128, 128, 128))
            >>> out = torch.reshape(torch.tensor(out).float(), (1, 1, 128, 128, 128))
            >>> ici_loss_function = ICILoss(
            >>>         loss_function_pixel=loss_dice,
            >>>         loss_function_instance=loss_dice,
            >>>         loss_function_center=loss_dice,
            >>>         num_out_chn = 1,
            >>>         activation="none",
            >>>     )
            >>> ici_loss_function.print_parameters()
            >>> seg_pixel, seg_instance, seg_center, fdr, cc_falsed, cc_missed = ici_loss_function(
            >>>         out,
            >>>         lbl
            >>>     )
        """
        
        # Dice loss function to calculate the rate of missed and false instances
        loss_dice_per_output_instance = DiceLoss(
            to_onehot_y=False, 
            sigmoid=False, 
            softmax=False
        )

        # Calculate non-linear function outside the model. Especially useful if ones use MONAI models
        # where non-linear function is (usually) not included within the models.
        if self.activation == "sigmoid":
            outputs_act = torch.sigmoid(outputs)
            mask = torch.ge(outputs_act, self.segmentation_threshold)
            outputs_act = outputs_act * mask.int()
            outputs_act[outputs_act > 0] = 1

        elif self.activation == "softmax" and self.num_out_chn > 1:
            outputs_act = torch.nn.functional.softmax(outputs, dim=1) # perform softmax on channel's dim (i.e., 1)
            num_all_classes = outputs_act.shape[1]
            outputs_act = torch.unsqueeze(outputs_act[:, self.object_chn, ...], dim=1)
            outputs_act = torch.ceil(outputs_act * torch.ge(outputs_act, 1/num_all_classes).float())

        elif self.activation == "none":
            outputs_act = outputs

        ## CALCULATE SEGMENTATION LOSS GLOBALLY
        if self.activation == "softmax" and self.num_out_chn > 1 and self.object_chn > 1:
            segmentation_loss = self.loss_function_pixel(outputs_act, labels)
            
        elif self.activation != "softmax" and self.num_out_chn == 1 and self.object_chn == 1:
            segmentation_loss = self.loss_function_pixel(outputs, labels)

        batch_length = labels.shape[0]

        ### FOR LOOP FOR ALL BATCHES
        batch_counter = 0
        for bid in range(batch_length):
            if bid < batch_length:
                end_index = bid + 1
            elif bid == batch_length:
                end_index = None

            # Flags for lists of all instances loss values
            instance_wise_loss_flag = False

            batch_counter += 1
            label_batch = labels[bid:end_index, ...]
            output_batch = outputs_act[bid:end_index, ...]

            if self.instance_wise_loss_no_tp:
                output_batch_noTP = (output_batch * (~label_batch.bool())).detach()
                # nim = nib.Nifti1Image(output_batch_noTP.squeeze().cpu().numpy().astype('float32'), np.eye(4))
                # nib.save(nim, 'blobs/output_batch_noTP.nii.gz')

            if cc_label_batch is None:
                # Connected component analysis for label
                cc_label = connected_components(
                    label_batch, 
                    num_iterations=self.num_iterations,
                    spatial_dims=self.spatial_dims
                    )
            else:
                cc_label = cc_label_batch[bid:end_index, ...]

            # Connected component analysis for output
            cc_output = connected_components_with_gradients(
                output_batch, 
                num_iterations=self.num_iterations,
                threshold=self.segmentation_threshold,
                spatial_dims=self.spatial_dims
                )

            # In case there are no label instances, we will return this value
            loss_batch = self.loss_function_instance(output_batch, label_batch)
            
            ## CREATE VARIABLES FOR INITIAL VALUES
            lbl_instance_centers = torch.zeros_like(label_batch)
            output_batch_centers = output_batch * 0.

            cc_label_unique_all = torch.unique(cc_label) 
            cc_label_unique = torch.masked_select(cc_label_unique_all, (cc_label_unique_all > 0))
            cc_label_unique_correct = torch.zeros_like(cc_label_unique, dtype=torch.bool)
            num_cc_label = len(cc_label_unique)

            cc_output_unique_all = torch.unique(cc_output)
            cc_output_unique = torch.masked_select(cc_output_unique_all, (cc_output_unique_all > 0))
            cc_output_unique_correct = torch.zeros_like(cc_output_unique, dtype=torch.bool)
            num_cc_output = len(cc_output_unique)

            ## AUXILIARY VARIABLES FOR DETECTION TERM
            ## IN CASE THERE ARE NO INSTANCES IN OUTPUT/INPUT IMAGE
            num_cc_output_temp = num_cc_output
            if num_cc_output < 1:
                num_cc_output_temp = 1
            num_cc_label_temp = num_cc_label
            if num_cc_label < 1:
                num_cc_label_temp = 1

            # Flag for no detection (output) instances
            no_detections = num_cc_output == 0 and num_cc_output_temp == 1

            # Flag that there are too many detection (output) instances. Used for efficiency.
            too_many = num_cc_label > 0 and num_cc_output > (num_cc_label * self.mul_too_many)

            ### COMPUTE AND FIND THE CENTERS OF THE INSTANCES FROM "OUTPUT" DATA (PREDICTED SEGMENTATION)
            cc_out_counter = 0
            bounding_boxes_output = False
            if num_cc_output > 0 and not too_many:  # If there are output instances (and not too many)
                for lbl in torch.unique(cc_output_unique):
                    cc_out_counter = cc_out_counter + 1

                    # Break if there are too many output instances
                    if self.max_cc_out != 0 and cc_out_counter > self.max_cc_out:
                        break

                    # find the location of the output instance
                    lbl_idx = cc_output == lbl  

                    # predicted segmentation with the output instance only
                    cc_out = torch.masked_select(output_batch, lbl_idx) 

                    # if the instance size is too small
                    # avoid too sensitive to the small instances
                    if cc_out.shape[0] <= self.min_instance_size:
                        continue

                    # find the min & max coordinates of the output instance
                    nonzero_idx = torch.nonzero(lbl_idx.int())
                    idx_min = torch.min(nonzero_idx, 0)[0]
                    idx_max = torch.max(nonzero_idx, 0)[0]

                    # Center of the mass of the output instance
                    idx_center = torch.ceil(torch.mean(nonzero_idx.float(), 0)).int()

                    # Get the corrected min & max coordinates of center of the mass.
                    # Used for the center of instance segmentation loss.
                    mins, maxs = get_corrected_indices(
                        idx_center, 
                        idx_center, 
                        self.centroid_offset, 
                        label_batch.shape,
                        spatial_dims=self.spatial_dims
                        )

                    # Create output max for the center of instance segmentation loss.
                    if self.spatial_dims == 3:
                        output_batch_centers[
                            0, 0, # batch, channel
                            mins[0]:maxs[0], # dim x
                            mins[1]:maxs[1], # dim y
                            mins[2]:maxs[2], # dim z
                        ] = 1. + self.smoother
                    elif self.spatial_dims == 2:
                        output_batch_centers[
                            0, 0, # batch, channel
                            mins[0]:maxs[0], # dim x
                            mins[1]:maxs[1], # dim y
                        ] = 1. + self.smoother

                    # Find if there are label instances that intersect with the output instance.
                    intersect_ids = torch.unique(torch.masked_select(cc_label, lbl_idx))
                    intersect_ids = torch.masked_select(intersect_ids, (intersect_ids > 0)) # exclude background
                    cc_label_location = torch.isin(cc_label_unique, intersect_ids)

                    # Boolean map to indicate which label instance intersect with output instance (initialization)
                    detection_map = torch.zeros_like(
                        cc_label_location, 
                        device=output_batch.device, 
                        dtype=torch.bool,
                        requires_grad=False
                    )
                    
                    # Instance-wise loss for each output instance (initialization)
                    cc_label_loss = torch.zeros_like(
                        cc_label_location, 
                        device=output_batch.device, 
                        dtype=output_batch.dtype,
                        requires_grad=True
                    )

                    ## FOR DEBUGGING
                    # print("")
                    # print("cc_out_counter: ", cc_out_counter)
                    # print("len(intersect_ids): ", len(intersect_ids))
                    # print("cc_label_location: ", cc_label_location)
                    # Instance-wise loss for each output instance (updating)
                    if len(intersect_ids) > 1:
                        for iid in intersect_ids:
                            loss_per_instance = loss_dice_per_output_instance(
                                output_batch * lbl_idx, label_batch * torch.isin(cc_label, iid))
                            cc_label_loc = torch.isin(cc_label_unique, iid)
                            cc_label_loss = cc_label_loss + (cc_label_loc * loss_per_instance)
                    else:
                        loss_per_instance = loss_dice_per_output_instance(
                            output_batch * lbl_idx, label_batch * torch.isin(cc_label, intersect_ids))
                        cc_label_loss = cc_label_location * loss_per_instance

                    # print("cc_label_loss (up): ", cc_label_loss)

                    # Boolean map to indicate which label instance intersect with output instance (updating)
                    if num_cc_label > 0:
                        cc_label_loss = cc_label_loss + (~cc_label_location)
                        detection_map = cc_label_location

                        ## A version where one output instance can only detect one label instance
                        # detection_map[torch.argmin(cc_label_loss)] = True 
                        # detection_map = detection_map * cc_label_location
                    else:
                        cc_label_location = torch.zeros(1, device=output_batch.device, dtype=torch.bool, requires_grad=False)
                        cc_label_loss = torch.zeros(1, device=output_batch.device, dtype=output_batch.dtype, requires_grad=True)
                        detection_map = torch.zeros(1, device=output_batch.device, dtype=torch.bool, requires_grad=False)
                        
                    # print("cc_label_loss (final): ", cc_label_loss, cc_label_loss.requires_grad)
                    # print("detection_map: ", detection_map, detection_map.requires_grad)

                    ## ADD BOUNDING BOX INFORMATION FROM OUTPUT INSTANCES
                    if bounding_boxes_output:
                        detection_map_all = torch.cat((detection_map_all, torch.unsqueeze(detection_map, 0)), 0)
                        output_intersection = torch.cat((output_intersection, torch.unsqueeze(cc_label_location, 0)), 0)
                        loss_per_output_instance = torch.cat((loss_per_output_instance, torch.unsqueeze(cc_label_loss, 0)), 0)
                    else:
                        bounding_boxes_output = True
                        detection_map_all = torch.unsqueeze(detection_map, 0)
                        output_intersection = torch.unsqueeze(cc_label_location, 0)
                        loss_per_output_instance = torch.unsqueeze(cc_label_loss, 0)

                del cc_out
            else:
                temp = num_cc_output_temp
                if too_many:
                    temp = 1

                ## BOUNDING BOX INFORMATION FROM OUTPUT INSTANCES
                bounding_boxes_output = True
                detection_map_all = torch.zeros((temp, num_cc_label_temp), device=output_batch.device, dtype=torch.bool)
                output_intersection = torch.zeros((temp, num_cc_label_temp), device=output_batch.device, dtype=torch.bool)
                loss_per_output_instance = torch.ones(
                    (temp, num_cc_label_temp), device=output_batch.device, dtype=output_batch.dtype, requires_grad=True)

            ## Find which instances in label image correctly detecting output instances
            detected_instances = torch.max(detection_map_all, dim=0)
            
            ## FOR DEBUGGING
            # print("")
            # print("detected_instances: ", detected_instances)
            # print("detection_map_all: \n", detection_map_all)
            # print(asda)

            ### COMPUTE AND FIND THE CENTERS OF THE INSTANCES FROM "LABEL" DATA
            cc_lbl_counter = 0
            bounding_boxes_label = False
            if num_cc_label > 0:
                
                for lbl in cc_label_unique:
                    cc_lbl_counter += 1

                    # Find the indices of the label's instance 
                    lbl_idx = cc_label == lbl

                    # Create the mask of the label's instance
                    cc_lbl = torch.masked_select(label_batch, lbl_idx)

                    # if the instance size is too small
                    # avoid too sensitive to the small instances
                    if cc_lbl.shape[0] <= self.min_instance_size:
                        continue

                    # Find the min & max coordinates of the output instance
                    nonzero_idx = torch.nonzero(lbl_idx.int())
                    idx_min = torch.min(nonzero_idx, 0)[0]
                    idx_max = torch.max(nonzero_idx, 0)[0]

                    # Find the center of mass of label's instance 
                    idx_center_lbl = torch.ceil(torch.mean(nonzero_idx.float(), 0)).int()

                    # Get the corrected min & max coordinates of center of the mass.
                    # Used for the center of instance segmentation loss.
                    mins, maxs = get_corrected_indices(
                        idx_center_lbl, 
                        idx_center_lbl, 
                        self.centroid_offset, 
                        label_batch.shape,
                        spatial_dims=self.spatial_dims
                        )
                        
                    # Create output max for the center of instance segmentation loss.
                    if self.spatial_dims == 3:
                        lbl_instance_centers[
                            0, 0, # batch, channel
                            mins[0]:maxs[0], # dim x
                            mins[1]:maxs[1], # dim y
                            mins[2]:maxs[2], # dim z
                        ] = 1. + self.smoother
                    elif self.spatial_dims == 2:
                        lbl_instance_centers[
                            0, 0, # batch, channel
                            mins[0]:maxs[0], # dim x
                            mins[1]:maxs[1], # dim y
                        ] = 1. + self.smoother

                    # Create label mask for the label's instance only
                    label_batch_cls = torch.zeros_like(label_batch)
                    label_batch_cls[lbl_idx] = torch.masked_select(label_batch, lbl_idx)

                    ## CALCULATION FOR INSTANCE WISE LOSS AND DETECTION LOSS (NEW FROM ICID Loss)
                    mask_out_instances = torch.zeros_like(label_batch, dtype=torch.bool)
                    if detected_instances[0][cc_lbl_counter-1]:

                        intersect_ind = torch.squeeze(torch.argwhere(detection_map_all[:,cc_lbl_counter-1]))
                        intersect_ids = cc_output_unique[intersect_ind.tolist()]
                        intersect_idx = torch.isin(cc_output, intersect_ids)
                        mask_out_instances[intersect_idx] = True

                    ## Case where the output instance with true positive is not used
                    if self.instance_wise_loss_no_tp:
                        mask_out_instances = mask_out_instances * (output_batch_noTP + label_batch_cls.bool())

                    ## CALCULATE SEGMENTATION OF THE INSTANCES
                    loss_per_instance = self.loss_function_instance(output_batch * mask_out_instances, label_batch_cls)

                    ## FOR DEBUGGING
                    ## Saving the results
                    # nim = nib.Nifti1Image((output_batch * mask_out_instances).squeeze().cpu().detach().numpy().astype('float32'), np.eye(4))
                    # nib.save(nim, 'blobs/lbl_' + str(cc_lbl_counter) + '_output.nii.gz')
                    # nim = nib.Nifti1Image(label_batch_cls.squeeze().cpu().detach().numpy().astype('float32'), np.eye(4))
                    # nib.save(nim, 'blobs/lbl_' + str(cc_lbl_counter) + '.nii.gz')

                    if self.weighted_instance_wise:
                        loss_per_instance_no_fp = self.loss_function_instance(
                            output_batch * mask_out_instances * label_batch_cls, label_batch_cls)
                        loss_per_instance = loss_per_instance + loss_per_instance_no_fp

                    ## ADD SEGMENTATION TO THE LIST FOR ALL INSTANCES
                    if instance_wise_loss_flag:
                        instance_wise_loss = torch.cat(
                            (instance_wise_loss, torch.unsqueeze(loss_per_instance, 0)), 0)
                    else:
                        instance_wise_loss = torch.unsqueeze(loss_per_instance, 0)
                        instance_wise_loss_flag = True

                    del cc_lbl, mask_out_instances, label_batch_cls, lbl_idx
            else:
                # In case there are no label instances
                instance_wise_loss = loss_batch.reshape(1,)
                instance_wise_loss_flag = True

            ## Calculate detection loss using Rate of Missed and False Instances (RMFI)
            loss_per_output_instance = torch.ceil((1 - loss_per_output_instance) * detection_map_all)
            correct_output_instances_sum = torch.sum(torch.max(loss_per_output_instance, dim=1)[0])
            detected_label_instances_sum = torch.sum(torch.max(loss_per_output_instance, dim=0)[0])
            instance_missed_loss = (num_cc_label - detected_label_instances_sum)

            # False positive calculation
            output_batch = output_batch * cc_out_counter
            num_of_false_instances = (num_cc_output - correct_output_instances_sum)
            if no_detections:
                num_of_false_instances = num_of_false_instances * 0.

            ## FOR DEBUGGING
            # print("")
            # print("detection_map_all: ", detection_map_all)
            # print("loss_per_output_instance: ", loss_per_output_instance)
            # print("")
            # print("num_cc_label: ", num_cc_label)
            # print("detected_label_instances_sum: ", detected_label_instances_sum)
            # print("instance_missed_loss: ", instance_missed_loss)
            # print("num_cc_output: ", num_cc_output)
            # print("correct_output_instances_sum: ", correct_output_instances_sum)
            # print("num_of_false_instances: ", num_of_false_instances)
            # print("")

            # Return RMFI instead of numbers of missed and false instances
            rate_of_false_detection = torch.abs(num_of_false_instances / (num_cc_output + self.smoother))
            rate_of_missed_detection = torch.abs(instance_missed_loss / (num_cc_label + self.smoother))
            if self.rate_instead_number:
                num_of_false_instances = rate_of_false_detection
                instance_missed_loss = rate_of_missed_detection

            # Calculate the false discovery rate (FDR) loss
            pred_sum = torch.sum(output_batch) + self.smoother
            pred__tp = torch.sum(output_batch * label_batch) + self.smoother
            ppv = pred__tp / pred_sum
            fdr = 1 - ppv

            if self.weighted_fdr:
                fdr = fdr * (num_of_false_instances + 1)

            ## Not needed, but the 'num_of_correct_detection' can be calculated with gradients
            if no_detections:
                cc_output_unique = cc_output_unique_all + 1
                cc_output_unique_correct = torch.zeros_like(cc_output_unique_all, dtype=torch.bool)
            cc_output_unique_normed = torch.div(cc_output_unique, cc_output_unique.max())
            correct_detection_cc_output = torch.masked_select(cc_output_unique_normed, cc_output_unique_correct)
            num_of_correct_detection = torch.sum(torch.ceil(correct_detection_cc_output)).detach()

            ## Calculate Center-of-Instance (CI) segmentation loss
            cc_loss = self.loss_function_center(output_batch_centers, lbl_instance_centers)

            if self.reduce == "mean":
                instance_wise_loss_reduce = torch.mean(instance_wise_loss)
            elif self.reduce == "sum":
                instance_wise_loss_reduce = torch.sum(instance_wise_loss)

            if batch_counter == 1:
                if self.instance_wise_reduce == "instance":
                    instance_wise_loss_batch = instance_wise_loss
                elif self.instance_wise_reduce == "data":
                    instance_wise_loss_batch = torch.unsqueeze(instance_wise_loss_reduce, 0)
                instance_center_loss_batch = torch.unsqueeze(cc_loss, 0)
                instance_false_loss_batch = torch.unsqueeze(fdr, 0)
                instance_false_instances_loss_batch = torch.unsqueeze(num_of_false_instances, 0)
                instance_missed_instances_loss_batch = torch.unsqueeze(instance_missed_loss, 0)
            else:
                if self.instance_wise_reduce == "instance":
                    instance_wise_loss_batch = torch.cat((instance_wise_loss_batch, instance_wise_loss), 0)
                elif self.instance_wise_reduce == "data":
                    instance_wise_loss_batch = torch.cat(
                        (instance_wise_loss_batch, torch.unsqueeze(instance_wise_loss_reduce, 0)), 0)
                instance_center_loss_batch = torch.cat(
                    (instance_center_loss_batch, torch.unsqueeze(cc_loss, 0)), 0)
                instance_false_loss_batch = torch.cat(
                    (instance_false_loss_batch, torch.unsqueeze(fdr, 0)), 0)
                instance_false_instances_loss_batch = torch.cat(
                    (instance_false_instances_loss_batch, torch.unsqueeze(num_of_false_instances, 0)), 0)
                instance_missed_instances_loss_batch = torch.cat(
                    (instance_missed_instances_loss_batch, torch.unsqueeze(instance_missed_loss, 0)), 0)

            del instance_wise_loss, instance_wise_loss_reduce, cc_loss, fdr, instance_missed_loss
            del num_of_false_instances, cc_output, cc_label, cc_output_unique, cc_label_unique
            del cc_output_unique_normed, cc_output_unique_all, cc_label_unique_all, loss_batch

        instance_wise_loss = torch.mean(instance_wise_loss_batch)
        instance_center_loss = torch.mean(instance_center_loss_batch)
        instance_false_loss = torch.mean(instance_false_loss_batch)
        instance_false_instances_loss = torch.mean(instance_false_instances_loss_batch)
        instance_missed_instances_loss = torch.mean(instance_missed_instances_loss_batch)

        del instance_wise_loss_batch, instance_center_loss_batch, instance_false_loss_batch
        del instance_false_instances_loss_batch, instance_missed_instances_loss_batch

        return segmentation_loss, instance_wise_loss, instance_center_loss, instance_false_loss, \
            instance_false_instances_loss, instance_missed_instances_loss

