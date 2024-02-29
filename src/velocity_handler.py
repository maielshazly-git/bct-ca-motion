import os
import csv
import numpy as np
import pandas as pd


class VelocityHandler:
    def __init__(self, input_file, output_parent_dir, frames_per_sec, frame_count_in_rise_time):
        self.input_file = input_file
        self.output_parent_dir = output_parent_dir
        self.frames_per_sec = frames_per_sec
        self.output_dir = ''
        self.frame_count_in_rise_time = frame_count_in_rise_time
        
        
    def get_output_dir(self):
        return self.output_dir
        
        
    def calc_distance(self, y_x_1, y_x_2):
        y1, x1 = y_x_1.split('_')
        y2, x2 = y_x_2.split('_')
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
         
        
    def calc_velocity(self):
        contr_df = pd.read_csv(self.input_file, header=None, keep_default_na=False) # keep_default_na is False to convert NaNs to empty strings
        contr_df_rows, contr_df_cols = contr_df.shape
                
        self.output_dir = os.path.join(self.output_parent_dir, 'output-velocities')
        os.makedirs(self.output_dir, exist_ok=True)
        
        csv_file = open(os.path.join(self.output_dir, "velocities.csv"), "a")
        csv_writer = csv.writer(csv_file)
        csv_row = [self.input_file]
        
        csv_file_max = open(os.path.join(self.output_dir, "max-velocities.csv"), "a")
        csv_writer_max = csv.writer(csv_file_max)
        
        csv_file_max_dist = open(os.path.join(self.output_dir, "max-distances.csv"), "a")
        csv_writer_max_dist = csv.writer(csv_file_max_dist)
        
        max_distance, max_speed = 0, 0
        for col in range(contr_df_cols):
            col_flows = list(contr_df[col])
            distance, last_frame_with_motion = 0, 0
            
            for row in range(1, self.frame_count_in_rise_time):
                prev_flow, curr_flow = col_flows[row - 1], col_flows[row]
                if curr_flow == '':
                    break
                    
                dist = self.calc_distance(prev_flow, curr_flow)
                if dist != 0:
                    last_frame_with_motion = row
                distance += dist
                
            distance /= 138 # converting to mm, 1mm of the BCT corresponds to an average of 138 pixels, so conversion is based on this
            seconds = (last_frame_with_motion + 1) / self.frames_per_sec
            col_speed = 0 if seconds == 0 else distance / seconds
            
            if distance > max_distance:
                max_distance = distance
                max_speed = col_speed
            csv_row.append(round(col_speed, 3))
            
        csv_writer.writerow(csv_row)
        csv_writer_max.writerow([self.input_file, round(max_speed, 3)])
        csv_writer_max_dist.writerow([self.input_file, round(max_distance, 3)])
        
        
        return round(max_speed, 3), "max-velocities.csv"
        
        
