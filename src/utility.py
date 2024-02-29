import os
import numpy as np
from PIL import Image


def get_file_name_from_dir(input_dir, frame_order=None):
    file_name = os.listdir(input_dir)[0] # get the 1st file name, any file is fine for now
    root_name, extension = file_name.split('.') # split name from extension so that frame order is the last element in name
    index_before_order = root_name.rfind('_')
    frame_order_str = root_name[index_before_order + 1:]
    
    root_name = root_name[:index_before_order + 1]
    
    if frame_order is None: # for OpenCV video capture
        root_name += '%d'
    else: # for masks
        root_name += str(frame_order)
    return os.path.join(input_dir, root_name + '.' + extension)
    
    
def get_bct_indices(mask_path):
    im = Image.open(mask_path)
    im = np.array(im).astype('float32')
    
    indices = np.argwhere(im == 255)
    
    # Swap columns, I need [col, row] not [row, col] to be accepted by calcOpticalFlowPyrLK.
    indices[:, [0, 1]] = indices[:, [1, 0]]
    
    # I need [[col, row]] instead of [col, row] to be accepted by calcOpticalFlowPyrLK.
    indices = indices[:, np.newaxis, :]
    
    return indices
    
    
def get_pixel_mean_for_bct_and_bg(im_array, indices):
    # im_array: image array, a grayscale image WxH with no channel dimension
    # indices:  array of of arrays of integer colum and row indices, example:
    #           indices = [[[183, 136]], [[302, 123]], ...], where 183 is a column index and 136 is a row index
    #           get_bct_indices(mask_path) can be used to retrieve indices
    
    rows, cols = im_array.shape
    im_array_flat = im_array.flatten()
    
    flat_indices = []
    for i in indices:
        i = i[0]
        flat_index = (i[1] * cols) + i[0]
        flat_indices.append(flat_index)
        
    bct = im_array_flat[flat_indices]
    bg = np.delete(im_array_flat, flat_indices)
    return round(bct.mean()), round(bg.mean())
    
    
def get_divisors(n):
    div = []
    for i in range(1, int(n / 2) + 1):
        if n % i == 0:
            div.append(i)
    return div
    
    
