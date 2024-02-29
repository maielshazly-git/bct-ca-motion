import os
import cv2
import torch
import numpy as np
from glob import glob
import imageio.v2 as imageio
from torch_em.model import UNet2d
from torch.utils.data import Dataset
from alive_progress import alive_bar


class BCTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = [imageio.imread(path) for path in image_paths]
        self.masks = [imageio.imread(path) for path in mask_paths]
        self.transform = transform
        
        
    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        
        crop_shape = (crop_r, crop_c)
        shape = image.shape
        if shape != crop_shape:
            assert image.ndim == mask.ndim == 2
            beg_r = shape[0] - crop_r
            beg_c = shape[1] - crop_c
            image, mask = image[beg_r:, beg_c:], mask[beg_r:, beg_c:]
            
        image, mask = np.array(image), np.array(mask)
        
        if self.transform is not None:
            image, mask = self.transform(image), self.transform(mask)
        # To add the channel dimension, or else I get syntax errors upon plotting samples
        if image.ndim == 2:
            image = image[None]
        if mask.ndim == 2:
            mask = mask[None]
        return image, mask
        
        
    def __len__(self):
        return len(self.images)
        
        
class MaskSegmenter:
    def __init__(self, input_dir, output_parent_dir, crop_r, crop_c, device='cuda'):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.output_dir = ''
        self.crop_r = crop_r # Original sizes are 270x360, so I chose the nearest sizes that are divisible by 16 in both dimensions
        self.crop_c = crop_c # Original sizes are 270x360, so I chose the nearest sizes that are divisible by 16 in both dimensions
        self.device = device
        
        
    def set_input_dir(self, input_dir):
        self.input_dir = input_dir
        
        
    def get_output_dir(self):
        return self.output_dir
        
        
    def dtype_transform(self, x):
        return x.astype('float32')
        
        
    def pred_test_im(self, im, model):
        # crop to a suitable size with the least assumed error, by cropping out parts in a way that can be easier compensated
        # instead of cropping out from one side (left or right), I crop in the horizontal middle so that I can have more predictions on both sides
        # I crop vertically all pixels from the top because BCTs don't exist there
        beg_r = im.shape[0] - self.crop_r
        beg_c = int((im.shape[1] - self.crop_c) / 2)
        im = im[beg_r:, beg_c:beg_c+self.crop_c][None, None] # add batch and channel dimensions or else a syntax error occurs upon passing to the model, so NCWH
        
        pred = torch.sigmoid(model(im))
        pred[pred != 1] = 0 # I did this to remove the boundary, since it had a visually different color/look, I checked that 1's are BCT from print(np.unique(pred))
        return pred.detach().cpu().numpy().squeeze(), beg_r, beg_c
        
        
    def expand_im_to_ori_size(self, im, beg_r, beg_c, pred):
        # pad top part (and no bottom part to be padded)
        top_rows = np.zeros((beg_r, self.crop_c)) # I'm sure the BCT is not at the very top, so I assign 0's to these pixels
        padded_pred = np.append(top_rows, pred, axis=0)
        # pad left part
        col = padded_pred[:, 0]
        col = col.reshape((col.size, 1))
        for i in range(beg_c):
            padded_pred = np.append(col, padded_pred, axis=1)
        # pad right part
        col = padded_pred[:, -1]
        col = col.reshape((col.size, 1))
        for i in range(beg_c):
            padded_pred = np.append(padded_pred, col, axis=1)
        return padded_pred
        
        
    def infer(self, checkpoint_dir, mask_generator=True):
        # mask_generator: The function works as a mask generator if this flag is True and works as a mask enhancer otherwise.
        checkpoint_file = os.path.join(checkpoint_dir, 'mask-generator.pt') if mask_generator else os.path.join(checkpoint_dir, 'mask-enhancer.pt')
        
        self.output_dir = os.path.join(self.output_parent_dir, 'masks')
        os.makedirs(self.output_dir, exist_ok=True)
        im_paths = glob(os.path.join(self.input_dir, '*'))
        
        with alive_bar(len(im_paths)) as bar:
        
            for path in im_paths:
            
                im = imageio.imread(path).astype('float32')
                im_name = path.split('/')[-1]
                
                model = UNet2d(in_channels=1, out_channels=1)
                
                if self.device == 'cpu':
                    im = torch.from_numpy(im)
                    model = torch.load(checkpoint_file, map_location=torch.device(self.device))
                else:
                    im = torch.from_numpy(im).to(self.device)
                    model = torch.load(checkpoint_file) # loaded to cuda by default
                    
                pred, beg_r, beg_c = self.pred_test_im(im, model)
                pred = self.expand_im_to_ori_size(im, beg_r, beg_c, pred)
                
                pred[pred == 1] = 255 # predictions are in the range [0, 1]
                
                if mask_generator:
                    cv2.imwrite(os.path.join(self.output_dir, 'Mask_' + im_name.split('.')[0] + '.png'), pred)
                else:
                    cv2.imwrite(os.path.join(self.output_dir, im_name.split('.')[0] + '.png'), pred)
                    
                bar()
                
                
