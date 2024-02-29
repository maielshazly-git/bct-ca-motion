import os
import glob
import numpy as np
from PIL import Image
from alive_progress import alive_bar
from utility import get_bct_indices, get_file_name_from_dir, get_pixel_mean_for_bct_and_bg


class Preprocessor:
    def __init__(self, input_dir, output_parent_dir):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.output_dir = ''
        
        
    def get_output_dir(self):
        return self.output_dir
        
        
    def set_output_dir_after_smoothing(self):
        # Because smooth_spatially_temporally_one_frame() doesn't permanently update self.output_dir due to calling it as a target in a Process.
        # So, make sure to set self.output_dir to the smoothing dir after finishing all Processes. This function should be called in the main part that calls the Processes.
        self.output_dir = os.path.join(self.output_parent_dir, 'spatially-temporally-smoothed')
        
        
    def normalize(self):
        # Originally, frames are not in range normalized in range [0, 1]. For instance, the unique values of the 1st frame are [21 22 23 24 25 26 27 28 29 30 31 32].
        min_vs, max_vs, = 255, 0
        im_paths = glob.glob(os.path.join(self.input_dir, '*'))
        
        print("Part 1/2... Computing global maximum and minimum")
        with alive_bar(len(im_paths)) as bar:
            for im_name in im_paths:
                im = Image.open(im_name)
                im = np.array(im, dtype=float)
                values = np.unique(im)
                
                min_, max_ = np.min(values), np.max(values)
                if min_ < min_vs:	min_vs = min_
                if max_ > max_vs:	max_vs = max_
                
                bar()
            
        self.output_dir = os.path.join(self.output_parent_dir, 'normalized')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Part 2/2... Normalizing frames")
        with alive_bar(len(im_paths)) as bar:
            for im_name in im_paths:
                im = Image.open(im_name)
                im = np.array(im, dtype=float)
                im = (im - min_vs) / (max_vs - min_vs)
                
                im_new = Image.fromarray(im)
                im_new.save(os.path.join(self.output_dir, im_name.split('/')[-1]))
                
                bar()
                
                
    def smooth_spatially_temporally_one_frame(self, frame_i, frame_count, kernel_r=3, kernel_c=3, kernel_t=4):
        # Calling normalize() is a prerequisite for calling this function.
        
        # This function smoothes one frame, which is the frame at order frame_i.
        # This function is expected to be used in a calling code that uses multiprocessing to allow calling it by multiple
        # parallel processes for different frames, depending on the number of available CPUs found.
        # This function requires a calling code with additional instructions to provide it the required frame to be processed.
        
        # The caller code of this function cannot be a function in this class because it needs to be in the main script to be
        # called on the terminal, due to if __name__ == "__main__".
        
        input_dir = self.output_dir # output directory of the normalization
        self.output_dir = os.path.join(self.output_parent_dir, 'spatially-temporally-smoothed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # The kernel spatial dimensions should be odd
        if kernel_r % 2 == 0:
            kernel_r -= 1
        if kernel_c % 2 == 0:
            kernel_c -= 1
            
        # Limit the spatial size of the kernel to 17x17, I don't leave it open
        kernel_r = min(kernel_r, 17)
        kernel_c = min(kernel_c, 17)
            
        frame_j = frame_i + 1
        im_name = get_file_name_from_dir(input_dir, frame_i + 1)
        image = Image.open(im_name)
        im = np.array(image, dtype=float)
        
        rows, cols = im.shape
        im_new = np.empty((rows, cols), dtype=float)
        
        pad_r, pad_c = int(kernel_r / 2), int(kernel_c / 2)
        
        if frame_i + kernel_t <= frame_count:
            time_steps = kernel_t
        else:
            time_steps = frame_count - frame_i
            
        while frame_j < (frame_i + time_steps):
            im_name_j = get_file_name_from_dir(input_dir, frame_j + 1)
            image = Image.open(im_name_j)
            im = np.add(im, np.array(image, dtype=float))
            frame_j += 1
            
        im = np.pad(im, pad_r, mode='constant') # assuming only square kernels, so pad_r or pad_c
        
        for row in range(rows):
            for col in range(cols):
                region = im[row:(row + kernel_r), col:(col + kernel_c)]
                im_new[row, col] = np.sum(region) / (region.size * time_steps)
                
        im_new = (im_new * 255).astype(np.uint8)
        
        im_new = Image.fromarray(im_new)
        im_new.save(os.path.join(self.output_dir, im_name.split('/')[-1]))
        
        
    def smooth_spatially_temporally(self, kernel_r=3, kernel_c=3, kernel_t=4):
        # Calling normalize() is a prerequisite for calling this function.
        
        # This function smoothes all frames one frame at a time, without multiprocessing.
        # This function is complete on its own, it just needs to be called.
        
        input_dir = self.output_dir # output directory of the normalization
        self.output_dir = os.path.join(self.output_parent_dir, 'spatially-temporally-smoothed')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # The kernel spatial dimensions should be odd
        if kernel_r % 2 == 0:
            kernel_r -= 1
        if kernel_c % 2 == 0:
            kernel_c -= 1
            
        # Limit the spatial size of the kernel to 17x17, I don't leave it open
        kernel_r = min(kernel_r, 17)
        kernel_c = min(kernel_c, 17)
        
        frame_names = glob.glob(os.path.join(input_dir, '*'))
        frame_count = len(frame_names)
        pad_r, pad_c = int(kernel_r / 2), int(kernel_c / 2)
        
        with alive_bar(frame_count) as bar:

            # This loop goes over the existing frames to form newly empty frames with the same dimensions and gathers them in a list of frames, i.e., one frame at a time.
            # images is for example of shape (6000, 270, 360).
            for frame_i in range(frame_count):

                # Manually form names since they aren't sorted in glob, and if I sort by name, they would be sorted as 1, 100, ..., 2, 200, ...
                im_name = get_file_name_from_dir(input_dir, frame_i + 1)
                
                if frame_i + kernel_t <= frame_count:
                    time_steps = kernel_t
                else: # take whatever frames remain; their count is less than kernel_t
                    time_steps = frame_count - frame_i
                    
                frame_j = frame_i + 1 # + 1 because I already consider the current frame in the following couple of lines
                image = Image.open(im_name)
                im = np.array(image, dtype=float)
                
                rows, cols = im.shape
                im_new = np.empty((rows, cols), dtype=float)
                
                # This loop goes over the original frames to work with kernels on them, i.e., time_steps frames at a time
                while frame_j < (frame_i + time_steps):
                    im_name_j = get_file_name_from_dir(input_dir, frame_j + 1)
                    image = Image.open(im_name_j)
                    im = np.add(im, np.array(image, dtype=float))
                    frame_j += 1
                    
                im = np.pad(im, pad_r, mode='constant') # assuming only square kernels, so pad_r or pad_c
        
                # These loops go over the pixels of the new frame to form their smoothed values
                for row in range(rows):
                    for col in range(cols):
                        region = im[row:(row + kernel_r), col:(col + kernel_c)]
                        im_new[row, col] = np.sum(region) / (region.size * time_steps)
                        
                im_new = (im_new * 255).astype(np.uint8)
                
                im_new = Image.fromarray(im_new)
                im_new.save(os.path.join(self.output_dir, im_name.split('/')[-1]))
                
                bar()
                
                
