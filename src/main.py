import os
import csv
import glob
import argparse
import multiprocessing
from multiprocessing import Process
from alive_progress import alive_bar
from preprocessor import Preprocessor
from mask_segmenter import MaskSegmenter
from fluorescence_signals_processor import FluorescenceSignalsProcessor
from motion_tracker import MotionTracker
from velocity_handler import VelocityHandler
from flows_averaging_handler import FlowsAveragingHandler


class Starter:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
        self.output_parent_dir = 'Output'
        os.makedirs(self.output_parent_dir, exist_ok=True)
        
        self.smoothed_images_dir = ''
        self.masks_dir = ''
        self.flows_output_dir = ''
        
        self.left_ips = []
        self.right_ips = []
        self.primary_peaks = []
        
        self.contractions_count = 0
        self.image_cols = 0
        self.flows_colors = []
        
        
    def preprocess(self, smooth_kernel_r, smooth_kernel_c, smooth_kernel_t, smooth_multiprocess, cpu_count):
        # If smooth_multiprocess is True and no cpu_count is provided, then the maximum number of CPUs found is used.
        
        print("Preprocessing... Normalization")
        preprocessor = Preprocessor(self.input_dir, self.output_parent_dir)
        preprocessor.normalize()
        
        print("Preprocessing... Spatio-temporal-smoothing")
        if not smooth_multiprocess:
            preprocessor.smooth_spatially_temporally()
        else:
            if __name__ == "__main__":
                max_cpu_count = multiprocessing.cpu_count()
                if cpu_count is None or cpu_count > max_cpu_count:
                    cpu_count = int(max_cpu_count / 2)
                
                frame_names = glob.glob(os.path.join(self.input_dir, '*'))
                frame_count = len(frame_names)
                
                frame_i = 0
                with alive_bar(frame_count) as bar:
                    while frame_i < frame_count:
                        processes = []
                        for cpu in range(cpu_count):
                            process = Process(
                                target=preprocessor.smooth_spatially_temporally_one_frame, args=(
                                    frame_i, frame_count, smooth_kernel_r, smooth_kernel_c, smooth_kernel_t))
                            processes.append(process)
                            process.start()
                            frame_i += 1
                            if frame_i == frame_count:
                                break
                                
                        for process in processes:
                            process.join()
                            bar()
                            
                preprocessor.set_output_dir_after_smoothing()
        self.smoothed_images_dir = preprocessor.get_output_dir()
        
        
    def segment(self, segment_with_cuda, enhance_masks_after_segmentation):
        device = 'cuda' if segment_with_cuda else 'cpu'
        
        crop_r, crop_c = 256, 352
        mask_segmenter = MaskSegmenter(self.input_dir, self.output_parent_dir, crop_r, crop_c, device)
        print("\nMask segmentation... Mask generation")
        mask_segmenter.infer('checkpoints', mask_generator=True)
        self.masks_dir = mask_segmenter.get_output_dir()
        
        if enhance_masks_after_segmentation:
            mask_segmenter.set_input_dir(self.masks_dir)
            print("Mask segmentation... Mask enhancement")
            mask_segmenter.infer('checkpoints', mask_generator=False)
            self.masks_dir = mask_segmenter.get_output_dir()
            
            
    def process_signals(self, smooth_kernel_t):
        fluorescence_signals_processor = FluorescenceSignalsProcessor(self.smoothed_images_dir, self.output_parent_dir, self.masks_dir)
        print("\Fluorescence signals processing...")
        
        ca_sig_expected_path = os.path.join(self.output_parent_dir, 'calcium-signal.csv')
        if os.path.exists(ca_sig_expected_path) and os.path.getsize(ca_sig_expected_path) > 0: # if calcium signal file already exists and is not empty, use it
            deltaf_over_f0 = list(csv.reader(open(ca_sig_expected_path, "r")))
            deltaf_over_f0 = [float(item) for sublist in deltaf_over_f0 for item in sublist]
            fluorescence_signals_processor.set_deltaf_over_f0(deltaf_over_f0)
            self.left_ips, self.right_ips, = fluorescence_signals_processor.get_bounding_frames_of_contractions(deltaf_over_f0)
            print("Complete")
            
        else:
            fluorescence_signals_processor.get_deltaf_over_f0(smooth_kernel_t)
            self.left_ips, self.right_ips = fluorescence_signals_processor.get_bounding_frames_of_contractions() # list not passed, uses its own computed list from the function above
            
        self.primary_peaks = fluorescence_signals_processor.get_frames_of_primary_peaks()
        print(self.primary_peaks)
                    
    def track_motion(self, time_bet_frame_present_msec):
        self.smoothed_images_dir = os.path.join(self.output_parent_dir, 'spatially-temporally-smoothed')
        self.masks_dir = os.path.join(self.output_parent_dir, 'masks')
        
        motion_tracker = MotionTracker(self.smoothed_images_dir, self.output_parent_dir, self.masks_dir, self.left_ips, self.right_ips)
        print("\nMotion tracking...")        
        self.contractions_count, self.image_cols, self.flows_colors = motion_tracker.lucas_kanade_method(time_bet_frame_present_msec)
        self.flows_output_dir = motion_tracker.get_flows_output_dir()
        
        
    def handle_velocity(self, frames_per_sec):
        print("\Velocity calculation...")
        with alive_bar(self.contractions_count) as bar:
            max_velocities = []
            
            for i in range(1, self.contractions_count + 1):
                contraction_file = os.path.join(self.flows_output_dir, 'contraction_' + str(i) + '.csv')
                
                frame_count_in_rise_time = self.primary_peaks[i - 1] - self.left_ips[i - 1] + 1
                velocity_handler = VelocityHandler(contraction_file, self.output_parent_dir, frames_per_sec, frame_count_in_rise_time)
                max_velocity, max_velocities_file = velocity_handler.calc_velocity()
                
                if max_velocity is None and max_velocities_file is None:
                    print("No contractions to calculate velocity for.")
                    return
                    
                max_velocities.append(max_velocity)
                bar()
            
        csv_file_max = open(os.path.join(velocity_handler.get_output_dir(), max_velocities_file), "a")
        csv_writer_max = csv.writer(csv_file_max)
        csv_writer_max.writerow([]) # just an empty row
        csv_writer_max.writerow(['Average', round(sum(max_velocities) / len(max_velocities), 3)])
        
        
    def average_flows(self, contraction_time_div, grid_div):
        print("\nFlows averaging...")
        with alive_bar(self.contractions_count) as bar:
        
            for i in range(1, self.contractions_count + 1):
                contraction_file = os.path.join(self.flows_output_dir, 'contraction_' + str(i) + '.csv')
                
                flows_averaging_handler = FlowsAveragingHandler(self.smoothed_images_dir, self.output_parent_dir, self.masks_dir, grid_div, self.image_cols)
                
                primary_peak = self.primary_peaks[i - 1] - self.left_ips[i - 1]
                flows_averaging_handler.summarize_flows_in_grid(contraction_file, contraction_time_div, self.flows_colors, self.left_ips[i - 1], primary_peak)
                bar()
                
                
parser = argparse.ArgumentParser()

required_args = parser.add_argument_group('required arguments')
required_args.add_argument('--input_dir', type=str, required=True, help='Directory of the input image sequence')
required_args.add_argument('--frames_per_sec', type=int, required=True, help='Number of frames taken per second in the input image sequence')

parser.add_argument('--smooth_kernel_row_steps', type=int, default=3, help='Height of the smoothing kernel in preprocessing, default is 3')
parser.add_argument('--smooth_kernel_col_steps', type=int, default=3, help='Width of the smoothing kernel in preprocessing, default is 3')
parser.add_argument('--smooth_kernel_time_steps', type=int, default=4, help='Time steps of the smoothing kernel in preprocessing, default is 4')
parser.add_argument('--smooth_multiprocess', action='store_true', help='If this flag is on, smoothing in preprocessing uses multiprocessing') # False if not provided
parser.add_argument('--cpu_count', type=int, default=None, help='Number of CPUs used by multiprocessing in smoothing')
parser.add_argument('--segment_with_cuda', action='store_true', help='If this flag is on, mask segmentation uses cuda, and CPU otherwise') # False if not provided
parser.add_argument('--enhance_masks_after_segmentation', action='store_true', help='If this flag is on, mask generation is followed by mask enhancement') # False if not provided
parser.add_argument('--time_bet_frame_present_msec', type=int, default=1, help='Speed of presenting frames in motion tracking (in milliseconds)')
parser.add_argument('--contraction_time_division', type=int, default=0, help='Number of time splits into which the duration of the signal is divided in flow-averaging. If not provided or 0, the signal is divided into rise and decay times.')
parser.add_argument('--grid_division', type=int, default=3, help='Number of columns into which the BCT is divided in flow-averaging')
args = parser.parse_args()

starter = Starter(args.input_dir)
starter.preprocess(args.smooth_kernel_row_steps, args.smooth_kernel_col_steps, args.smooth_kernel_time_steps, args.smooth_multiprocess, args.cpu_count)
starter.segment(args.segment_with_cuda, args.enhance_masks_after_segmentation)
starter.process_signals(args.smooth_kernel_time_steps)
starter.track_motion(args.time_bet_frame_present_msec)
starter.handle_velocity(args.frames_per_sec)
starter.average_flows(args.contraction_time_division, args.grid_division)

