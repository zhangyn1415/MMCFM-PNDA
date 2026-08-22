import cv2
import numpy as np
from scipy.ndimage import measurements
from skimage.segmentation import watershed
from scipy import ndimage

def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided. 

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel. 
    
    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out

def improved_segmentation(blb_raw, pred, marker):
    # Convert inputs to the required numeric types.
    pred = np.array(pred, dtype=np.float32)
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 0]
    
    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    # Compute the gradient magnitude directly from the direction maps.
    h_dir = cv2.normalize(h_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    # Use a compact Sobel kernel to preserve local boundaries.
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=3)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the normalized gradient magnitude.
    gradient_magnitude = np.sqrt(sobelh**2 + sobelv**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    # Strong gradients form watershed ridges; weak gradients form basins.
    distance_map = 1.0 - gradient_magnitude
    distance_map = distance_map * blb
    
    # Enhance instance separation with a foreground distance transform.
    from scipy import ndimage
    binary_blb = blb.astype(np.uint8)
    dist_transform = ndimage.distance_transform_edt(binary_blb)
    dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    
    # Combine boundary gradients with the foreground distance transform.
    combined_dist = 0.7 * distance_map + 0.3 * dist_transform
    
    # Watershed markers must use a signed integer type.
    if marker.dtype != np.int32:
        marker = marker.astype(np.int32)
    
    # Apply marker-controlled watershed segmentation.
    labels = watershed(-combined_dist, markers=marker, mask=blb.astype(bool))

    
    return labels

def process(pred_map, nr_types=None, threshold_=0.45, point=None):

    pred_type = pred_map[..., :-2]
    pred_inst = pred_map[..., -2:]
    pred_inst = np.squeeze(pred_inst)
    pred_all = []   
    
    # Assign each pixel to the most likely foreground class.
    for each_type in range(0, pred_type.shape[-1]-1):
        tensor = pred_type[:,:,each_type]
        nuclei_type = np.argmax(pred_type, axis=-1)
        tensor = np.where(nuclei_type==each_type, 1, 0)
        pred_all.append(tensor)
    
    pred_type = np.stack(pred_all, axis=-1)
    
    pred_type = pred_type.sum(axis=-1).astype(np.int32)
    pred_type = np.clip(pred_type, 0, 1)

    kk = np.cumsum(np.max(point, axis=0)).reshape((256, 256))
    pred_inst = improved_segmentation(pred_type, np.flip(pred_inst, axis=-1), np.max(point, axis=0))
    
    
    pred_inst_type = np.stack([np.where(one_cls > 0, pred_inst, 0) for one_cls in pred_all], axis=-1)
    
    return pred_inst_type
