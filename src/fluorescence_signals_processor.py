import os
import csv
import glob
import numpy as np
from PIL import Image
from alive_progress import alive_bar
from scipy.signal import find_peaks, peak_widths, peak_prominences
from utility import get_bct_indices, get_file_name_from_dir, get_pixel_mean_for_bct_and_bg


class FluorescenceSignalsProcessor:
    def __init__(self, input_dir, output_parent_dir, masks_dir):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.masks_dir = masks_dir
        self.deltaf_over_f0 = []
        self.left_ips = []
        self.right_ips = []
        self.peaks = []
        
        
    def set_deltaf_over_f0(self, deltaf_over_f0):
        self.deltaf_over_f0 = deltaf_over_f0
        
        
    def get_deltaf_over_f0(self, smooth_kernel_t=3):
        frame_names = glob.glob(os.path.join(self.input_dir, '*'))
        frame_count = len(frame_names) - (smooth_kernel_t - 1) # to remove edge cases
        frame_count = int(frame_count)
        
        f_minus_bg = []
        with alive_bar(frame_count) as bar:
        
            for frame_i in range(frame_count):
                # Manually form names since they aren't sorted in glob, and if I sort by name, they would be sorted as 1, 100, ..., 2, 200, ...
                im_name = get_file_name_from_dir(self.input_dir, frame_i + 1)
                image = Image.open(im_name)
                im = np.array(image, dtype=float)
                bct_indices = get_bct_indices(get_file_name_from_dir(self.masks_dir, frame_i + 1))
                
                f, background = get_pixel_mean_for_bct_and_bg(im, bct_indices)
                f_minus_bg.append(f - background)
                
                bar()
            
        f0, deltaf, self.deltaf_over_f0 = 0, [], []
        f0 = min(f_minus_bg)
        
        deltaf = [x - f0 for x in f_minus_bg]
        self.deltaf_over_f0 = [round(x / f0, 2) for x in deltaf]
        
        with open(os.path.join(self.output_parent_dir, "calcium-signal.csv"), "w") as cal_file:
            writer = csv.writer(cal_file)
            values = np.array(self.deltaf_over_f0)
            values = values.reshape(len(self.deltaf_over_f0), 1)
            writer.writerows(values)
            
            
    def get_bounding_frames_of_contractions(self, deltaf_over_f0=None):
        # deltaf_over_f0: By default, this class is expected to compute the deltaf_over_f0 list in get_deltaf_over_f0() and work on the attribute list in this function.
        # 		  But here, I allow the user to give a list to work on without having to depend on get_deltaf_over_f0(), in case a csv file with the list already exists.
        
        if deltaf_over_f0 is None:
            deltaf_over_f0 = self.deltaf_over_f0
            
        peak_widths_rel_height = 0.95
        
        peaks, _ = find_peaks(deltaf_over_f0)
        prominences = peak_prominences(deltaf_over_f0, peaks)[0]
        
        # If the max peak found is at vertical distance < 0.5, this indicates a non-contracting drug because this is not considered a contraction.
        if max(prominences) < 0.532: # threshold for non-contracting drugs
            return None, [min(1000, len(deltaf_over_f0))] # 1000 is just a number of frames to show that there are no flows, any reasonable number can be fine
            
        peaks, _ = find_peaks(deltaf_over_f0, prominence=0.532) # this value is at mean+3std of the noise
        widths = peak_widths(deltaf_over_f0, peaks, rel_height=peak_widths_rel_height)
        # widths[0]: widths
        # widths[1]: width_heights
        # widths[2]: left_ips
        # widths[3]: right_ips
        left_ips = np.floor(widths[2]).astype(int)
        right_ips = np.floor(widths[3]).astype(int)
        
        # Should sort first because in some cases of double- or triple- peaks, widths become not sorted and as a result,
        # I can't catch the overlapping ranges.
        combined = list(zip(left_ips, right_ips))
        sorted_with_left_ips = sorted(combined, key=lambda x: x[0])
        left_ips, right_ips = zip(*sorted_with_left_ips)
        left_ips, right_ips = list(left_ips), list(right_ips)
        
        i = 0
        while i < len(left_ips) - 1:
            update_i = True
            # Remove duplicates. I didn't use np.unique() because I want to make sure I remove the corresponding element in right_ips.
            if left_ips[i] in left_ips[:i]:
                update_i = False
                left_ips = np.delete(left_ips, i)
                right_ips = np.delete(right_ips, i)
                
            # Remove duplicates.
            if right_ips[i] in right_ips[:i]:
                update_i = False
                left_ips = np.delete(left_ips, i)
                right_ips = np.delete(right_ips, i)
                
            # Remove overlapping ranges occurring due to double-peaks. For example, if left_ips = [10, 15] and right_ips = [20, 40],
            # then return only left_ips = [10] and right_ips = [40]. It can be considered as merging the two ranges.
            if right_ips[i] > left_ips[i + 1]:
                update_i = False
                if left_ips[i] < left_ips[i + 1]: # I want to keep the minimum left
                    left_ips = np.delete(left_ips, i + 1)
                else:
                    left_ips = np.delete(left_ips, i)
                if right_ips[i] < right_ips[i + 1]: # I want to keep the maximum right
                    right_ips = np.delete(right_ips, i)
                else:
                    right_ips = np.delete(right_ips, i + 1)
                    
            if update_i:
                i += 1
                
        self.left_ips = left_ips
        self.right_ips = right_ips
        self.peaks = peaks
        return left_ips, right_ips
        
        
    def get_frames_of_primary_peaks(self):
        if self.left_ips is None: # if no contractions occurred, i.e., no peaks according to prominence threshold
            return
        primary_peaks = []
        for i in range(len(self.left_ips)):
            peak_frame_ca_intensity, peak_frame = -1, 0
            for peak in self.peaks:
                if self.left_ips[i] <= peak <= self.right_ips[i]:
                    if self.deltaf_over_f0[peak] > peak_frame_ca_intensity: # searching for the highest peak in contraction
                        peak_frame_ca_intensity = self.deltaf_over_f0[peak]
                        peak_frame = peak
            primary_peaks.append(peak_frame)
        return primary_peaks
        
        
