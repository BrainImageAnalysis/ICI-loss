import torch, math
import torch.nn.functional as F

## Adapted from: See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/connected_components.html>`
def connected_components(
        image: torch.Tensor, 
        num_iterations: int = 75,
        spatial_dims: int = 3
        ) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input imagetype is not a torch.Tensor. Got: {type(image)}")

    if not isinstance(num_iterations, int) or num_iterations < 1:
        raise TypeError("Input num_iterations must be a positive integer.")

    if len(image.shape) != (spatial_dims + 2):
        raise ValueError(
            f"Input image.shape (got: {image.shape}) must be equal to 'spatial_dims + 2' (got: {spatial_dims + 2})")

    if spatial_dims == 3:
        H, W, Z = image.shape[-3:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W, Z), dtype=torch.bool, device=image.device)
        mask[image == 1] = True

        ## allocate the output tensors for labels
        ## ORIGINAL
        # B, _, _, _, _ = image.shape
        # out = torch.arange(B * H * W * Z, device=image.device, dtype=image.dtype).view((-1, 1, H, W, Z)) 
        B = image.shape[0]
        out = torch.arange(B * H * W * Z, device=image.device, dtype=image.dtype).view(-1, 1, H, W, Z)
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)[mask]
    elif spatial_dims == 2:
        H, W = image.shape[-2:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W), dtype=torch.bool, device=image.device)
        mask[image == 1] = True

        ## allocate the output tensors for labels
        ## ORIGINAL
        # B, _, _, _, _ = image.shape
        # out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
        B = image.shape[0]
        out = torch.arange(B * H * W, device=image.device, dtype=image.dtype).view(-1, 1, H, W)
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return out.view_as(image)

def connected_components_with_gradients(
        image: torch.Tensor, 
        num_iterations: int = 75,
        threshold: float = 0.5,
        spatial_dims: int = 3
        ) -> torch.Tensor:

    if spatial_dims == 3:
        H, W, Z = image.shape[-3:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W, Z), dtype=torch.bool, device=image.device)
        mask[image >= threshold] = True
        image = torch.ceil(image * mask)
        
        B = image.shape[0]
        out = torch.arange(B * H * W * Z, device=image.device, dtype=image.dtype).view(-1, 1, H, W, Z)
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)[mask]

    elif spatial_dims == 2:
        H, W = image.shape[-2:]

        ## precompute a mask with the valid values
        mask = torch.zeros((1, 1, H, W), dtype=torch.bool, device=image.device)
        mask[image >= threshold] = True
        image = torch.ceil(image * mask)
        
        B = image.shape[0]
        out = torch.arange( B * H * W, device=image.device, dtype=image.dtype).view((-1, 1, H, W))
        out[~mask] = 0.

        for _ in range(num_iterations):
            out[mask] = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)[mask]

    return image * out

def get_corrected_indices(mins, maxs, offset, max_index, spatial_dims=3):
    # find the value of largest axis
    max_resolution = max(max_index[2:])
    
    offset_x = math.ceil(max_index[2] / max_resolution * offset)
    offset_y = math.ceil(max_index[3] / max_resolution * offset)
            
    min_x = mins[2]-offset_x-1
    min_y = mins[3]-offset_y-1
    max_x = maxs[2]+offset_x
    max_y = maxs[3]+offset_y

    min_x = 0 if min_x < 0 else min_x
    min_y = 0 if min_y < 0 else min_y

    max_x = max_index[2] if max_x > max_index[2] else max_x
    max_y = max_index[3] if max_y > max_index[3] else max_y

    if spatial_dims == 3:
        offset_z = math.ceil(max_index[4] / max_resolution * offset)
        min_z = mins[4]-offset_z-1
        max_z = maxs[4]+offset_z
        min_z = 0 if min_z < 0 else min_z
        max_z = max_index[4] if max_z > max_index[4] else max_z
        return [min_x, min_y, min_z], [max_x, max_y, max_z]
    elif spatial_dims == 2:
        return [min_x, min_y], [max_x, max_y]

        
def get_bbox_indices(mins, maxs, max_index, spatial_dims=3):
    offset = maxs - mins
    
    offset_x = offset[2]
    if offset_x == 0:
        offset_x += 1
    min_x = mins[2]-offset_x-1
    max_x = maxs[2]+offset_x

    offset_y = offset[3]
    if offset_y == 0:
        offset_y += 1
    min_y = mins[3]-offset_y-1
    max_y = maxs[3]+offset_y

    min_x = 0 if min_x < 0 else min_x
    min_y = 0 if min_y < 0 else min_y

    max_x = max_index[2] if max_x > max_index[2] else max_x
    max_y = max_index[3] if max_y > max_index[3] else max_y

    # center_x = torch.floor((max_x - min_x) / 2)
    # center_y = torch.floor((max_y - min_y) / 2)

    if spatial_dims == 3:
        offset_z = offset[4]
        if offset_z == 0:
            offset_z += 1
        min_z = mins[4]-offset_z-1
        max_z = maxs[4]+offset_z
        min_z = 0 if min_z < 0 else min_z
        max_z = max_index[4] if max_z > max_index[4] else max_z
        # center_z = torch.floor((max_z - min_z) / 2)
        return [min_x, min_y, min_z], [max_x, max_y, max_z]#, [center_x, center_y, center_z]
    elif spatial_dims == 2:
        return [min_x, min_y], [max_x, max_y]#, [center_x, center_y]

## BASED ON: https://github.com/pytorch/vision/blob/a192c95e77a4a4de3a8aeee45130ddc4d2773a83/torchvision/ops/boxes.py
def distance_box_iou_3D(center_output, min_output, max_output, center_label, min_label, max_label, smoother=1e-7):
    ## Calculate IoU based on the boxes information
    volume_output = (max_output[:,0] - min_output[:,0]) \
        * (max_output[:,1] - min_output[:,1]) \
        * (max_output[:,2] - min_output[:,2])
    volume_label = (max_label[:,0] - min_label[:,0]) \
        * (max_label[:,1] - min_label[:,1]) \
        * (max_label[:,2] - min_label[:,2])

    lt = torch.max(min_output[:, None, :], min_label[:, :])
    rb = torch.min(max_output[:, None, :], max_label[:, :])
    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2] # [N,M]
    union = volume_output[:, None] + volume_label - inter
    iou = inter / union

    x_p = center_output[:,0]
    y_p = center_output[:,1]
    z_p = center_output[:,2]
    x_g = center_label[:,0]
    y_g = center_label[:,1]
    z_g = center_label[:,2]
    
    centers_distance_squared = (
        (x_p[:, None] - x_g[None, :]) ** 2) + ((y_p[:, None] - y_g[None, :]) ** 2) + ((z_p[:, None] - z_g[None, :]) ** 2
    )
    
    lti = torch.min(min_output[:, None, :], min_label[:, :])  # [N,M,3]
    rbi = torch.max(max_output[:, None, :], max_label[:, :])  # [N,M,3]
    whi = (rbi - lti).clamp(min=0)  # [N,M,3]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + (whi[:, :, 2] ** 2) + smoother
    
    c_term = (centers_distance_squared / diagonal_distance_squared)

    # diou = 1 - iou - (centers_distance_squared / diagonal_distance_squared) # pytorch version
    diou = 1 - iou + (centers_distance_squared / diagonal_distance_squared)

    return diou, iou, c_term, centers_distance_squared

#### BASED ON: https://github.com/vqdang/hover_net/blob/master/models/hovernet/utils.py 
def mse_loss(true, pred):
    """Calculate mean squared error loss.
    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps 
    
    Returns:
        loss: mean squared error
    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss

####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of 
    horizontal and vertical map predictions. Assumes 
    channel 0 is Vertical and channel 1 is Horizontal.
    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps 
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)
    
    Returns:
        loss:  mean squared error of gradients
    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss

def HoVerRad3D(mask, background_xyz=0, background_radial=0):
    '''
    mask ~ [1, 1, H, W, D]
    returns:
    coords_x, coords_y, radial ~ [1, 1, H, W, D]
    '''
    dims = (mask.shape[2], mask.shape[3], mask.shape[4])

    coords = [torch.linspace(-1, 1, i) for i in dims]
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords, dim=len(dims))

    coords_x = coords[:, :, :, 0].to(mask.device)
    coords_y = coords[:, :, :, 1].to(mask.device)
    coords_z = coords[:, :, :, 2].to(mask.device)

    radial = torch.sqrt(coords_x*coords_x + coords_y*coords_y + coords_z*coords_z)
    thres = torch.nn.Sequential(torch.nn.Threshold(-0.5, 1), torch.nn.Threshold(0.5, 0))

    mask_threshold = thres(-mask).float()
    edge = F.max_pool3d(mask_threshold, 3, stride=1, padding=1) - mask_threshold

    x_min, x_max = coords_x[mask_threshold[0,0,...]>0.5].min(), coords_x[mask_threshold[0,0,...]>0.5].max()
    y_min, y_max = coords_y[mask_threshold[0,0,...]>0.5].min(), coords_y[mask_threshold[0,0,...]>0.5].max()
    z_min, z_max = coords_z[mask_threshold[0,0,...]>0.5].min(), coords_z[mask_threshold[0,0,...]>0.5].max()

    coords_x = 2*(coords_x - x_min)/(x_max-x_min) - 1
    coords_y = 2*(coords_y - y_min)/(y_max-y_min) - 1
    coords_z = 2*(coords_z - z_min)/(z_max-z_min) - 1

    radial = torch.sqrt(coords_x*coords_x + coords_y*coords_y + coords_z*coords_z)
    mean_radial = radial[edge[0,0,...]>0.5].mean()
    # print("mean_radial: ", mean_radial)
    radial = radial / mean_radial
    radial[mask_threshold[0,0,...]<0.5] = background_radial

    coords_x[mask_threshold[0,0,...]<0.5] = background_xyz
    coords_y[mask_threshold[0,0,...]<0.5] = background_xyz
    coords_z[mask_threshold[0,0,...]<0.5] = background_xyz

    coords_x = coords_x.unsqueeze(0).unsqueeze(0)
    coords_y = coords_y.unsqueeze(0).unsqueeze(0)
    coords_z = coords_z.unsqueeze(0).unsqueeze(0)

    radial = radial.unsqueeze(0).unsqueeze(0)

    return coords_x, coords_y, coords_z, radial