import os
import cv2
import math
import csv
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import circmean
from utility import get_bct_indices, get_file_name_from_dir


class FlowsAveragingHandler:
    def __init__(self, input_dir, output_parent_dir, masks_dir, grid_div, image_cols):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.masks_dir = masks_dir
        self.grid_div = grid_div
        self.image_cols = image_cols
        
        
    def get_big_grid_cells(self):
        # np.where which is used to get bct_indices goes over the image row by row, then
        # the columns in each row, as in for row for col in row. So, just get the min and
        # max rows from the first and last elements of bct_indices.
        bct_indices = get_bct_indices(get_file_name_from_dir(self.masks_dir, 1))
        
        min_row, max_row = bct_indices[0][0][1], bct_indices[-1][0][1]
        mid_row = (max_row - min_row) / 2
        mid_row = min_row + int(mid_row)
        
        col, grid_cols = 0, [0] # [0] for the left column of the first grid cell
        for i in range(self.grid_div - 1):
            col += int(self.image_cols / self.grid_div)
            grid_cols.append(col)
        grid_cols.append(self.image_cols) # for the right column of the last grid cell
        
        return [min_row, mid_row, max_row], grid_cols
        
        
    def draw_big_grid_cells(self, image, grid_rows, grid_cols, output_image_path):
        for i in range(1, len(grid_cols) - 1):
            image = cv2.line(image, (grid_cols[i], grid_rows[0]), (grid_cols[i], grid_rows[2]), (0, 0, 0), thickness=2)
            
        image = cv2.line(image, (0, grid_rows[1]), (grid_cols[-1] - 1, grid_rows[1]), (0, 0, 0), thickness=2)
        
        im_new = Image.fromarray(image)
        im_new.save(output_image_path)
        
        
    def get_contraction_time_splits(self, contraction_time_div, total_time_in_frames, primary_peak):
        splits = [0] # [0] for the beginning of the contraction
        
        if contraction_time_div <= 0:
            splits.append(primary_peak + 1) # + 1 to make the peak frame itself included in the first group
        else:
            frame = 0
            for i in range(contraction_time_div - 1):
                frame += int(total_time_in_frames / contraction_time_div)
                splits.append(frame)
    
        splits.append(total_time_in_frames - 1)
        return splits
        
        
    def bound_out_point_of_line_onto_cell_boundary(self, in_x, in_y, out_x, out_y, cell_x_beg, cell_y_beg, cell_x_end, cell_y_end):
        # (in_x, in_y):		Some point inside the cell or on its boundary, i.e., an accepted point.
        # (out_x, out_y):		Some point outside the cell, i.e., the point which needs fixing.
        # (cell_x_beg, cell_y_beg):	The top left point of the cell.
        # (cell_x_end, cell_y_end):	The bottom right point of the cell.
        
        if out_x > cell_x_end: # out of cell bounds to the right
            m = (out_y - in_y) / (out_x - in_x)
            c = in_y - (m * in_x)
            
            out_x = cell_x_end
            out_y = (m * out_x) + c
            
        if int(out_x) < cell_x_beg: # out of cell bounds to the left
            m = (out_y - in_y) / (out_x - in_x)
            c = in_y - (m * in_x)
            
            out_x = cell_x_beg
            out_y = (m * out_x) + c
            
        if out_y > cell_y_end: # out of cell bounds downwards
            # Handle this case since m in the line eq. would divide by delta_x = 0.
            if out_x == in_x:
                out_y = cell_y_end
            else:
                m = (out_y - in_y) / (out_x - in_x)
                c = in_y - (m * in_x)
                
                out_y = cell_y_end
                out_x = (out_y - c) / m
            
        if out_y < cell_y_beg: # out of cell bounds upwards
            # Handle this case since m in the line eq. would divide by delta_x = 0.
            if out_x == in_x:
                out_y = cell_y_beg
            else:
                m = (out_y - in_y) / (out_x - in_x)
                c = in_y - (m * in_x)
                
                out_y = cell_y_beg
                out_x = (out_y - c) / m
                
        return out_x, out_y
        
        
    def summarize_flows_in_grid_cell(self, image, df, flows_colors, frame_beg, frame_end, cell_x_beg, cell_x_end, cell_y_beg, cell_y_end, output_image_path, contraction_id, bounding_frames, cell_order):
        df_rows, df_cols = df.shape
        
        lengths_sum, flow_count = 0.0, 0
        beg_x_sum, beg_y_sum = 0.0, 0.0
        cos_sum, sin_sum = 0.0, 0.0 # for circular mean for average angle calculation
        angles_in_rad = []
        
        for col in range(df_cols):
            beg_point_status, end_point_status = '', ''
            passes_through_cell = [] # when I fix the outside beg/end point, I need a point that is inside the cell; this is how the fixing function works.
            points = [] # to draw subflows in the background
            
            # The inner loop sets beg and end points while considering the following cases of the current point df[col][row]:
            
            # 1. NaN		and	beg point not set
            # __ Flow is not existent in this time period, i.e., NaN was found before setting beg point.
            # 2. NaN		and	beg point is set
            # __ Only a point is found in the excel sheet.
            # __ Can be that there was a flow in a previous time period but wasn't detected anymore in this period.
            # 3. Not NaN	and	beg point not set	and	point is in cell range
            # 4. Not NaN	and	beg point is set	and	point is in cell range
            # 5. Not NaN	and	beg point not set	and	point not in cell range
            # __ No need to write a condition for this case.
            # __ The inner loop would simply end without chaning beg_point_status, so no flow is detected.
            # 6. Not NaN	and	beg point is set	and	point not in cell range
            # __ In this case, the beg point is in the current cell but the end point is somewhere outside it.
            # __ Handle this by considering the subflow present in the current cell.
            # __ So, compute a point in the same flow direction that ends on the boundary of the current cell.
            
            for row in range(frame_beg, frame_end + 1):
                # Is NaN
                if pd.isnull(df[col][row]):
                    break
                    
                # Not NaN
                elif not pd.isnull(df[col][row]):
                    x, y = df[col][row].split('_')
                    x, y = float(x), float(y)
                    
                    # Point belongs to the current cell
                    if cell_x_beg <= x <= cell_x_end and cell_y_beg <= y <= cell_y_end:
                        points.append([x, y])
                        passes_through_cell = [x, y] # any point that is inside the cell in this flow is needed
                        
                        if beg_point_status == '':
                            beg_x, beg_y = x, y
                            beg_point_status = 'set'
                            
                        # Won't break after setting in order to go on updating the end point until:
                        # 1. Either the end of the time period is reached, i.e., the inner loop ends.
                        # 2. Or a point which is not in the cell range is met.
                        else:
                            end_x, end_y = x, y
                            end_point_status = 'set'
                    # Here only for case 6 above, because in order to compute a point on the cell boundary, I need
                    # the line equation y = mx + c, and m = (y2 - y1) / (x2 - x1).
                    # Thus, I need a second point on the flow even if it's outside the cell.
                    else:
                        if beg_point_status == '':
                           beg_x, beg_y = x, y
                           beg_point_status = 'set-outside-cell'
                        else:
                           end_x, end_y = x, y
                           end_point_status = 'set-outside-cell'
            # End inner loop
            
            if beg_point_status != '' and end_point_status != '' and passes_through_cell != []:
                # Fixing point which is outside cell bounds
                if beg_point_status == 'set-outside-cell':
                    beg_x, beg_y = self.bound_out_point_of_line_onto_cell_boundary(passes_through_cell[0], passes_through_cell[1], beg_x, beg_y, cell_x_beg, cell_y_beg, cell_x_end, cell_y_end)
                    points.insert(0, [beg_x, beg_y])
                if end_point_status == 'set-outside-cell':
                    end_x, end_y = self.bound_out_point_of_line_onto_cell_boundary(passes_through_cell[0], passes_through_cell[1], end_x, end_y, cell_x_beg, cell_y_beg, cell_x_end, cell_y_end)
                    points.append([end_x, end_y])
                # End fixing point which is outside cell bounds
                
                # Checking if flow is not a point; otherwise it would affect flow_count incorrectly
                if not (beg_x == end_x and beg_y == end_y):
                    x1, y1 = points[0]
                    mask = np.zeros_like(image)
                    
                    color = flows_colors[col]
                    r, g, b = color[0], color[1], color[2]
                    r, g, b = int(r), int(g), int(b)
                    
                    for i in range(1, len(points)):
                        x2, y2 = points[i]
                        mask = cv2.line(mask, (int(x2), int(y2)), (int(x1), int(y1)), (r, g, b), thickness=4)
                        x1, y1 = x2, y2
                    mask = cv2.circle(mask, (int(x2), int(y2)), 5, (r, g, b), -1)
                    image = cv2.add(mask, image)
                    
                    beg_x_sum += beg_x
                    beg_y_sum += beg_y
                    
                    delta_x, delta_y = end_x - beg_x, end_y - beg_y
                    length = math.sqrt((delta_x)**2 + (delta_y)**2)
                    lengths_sum += length
                    
                    # Note: I tried out and atan2 already differentiates between 0 and 180
                    angle_in_rad = math.atan2(delta_y, delta_x)
                    if angle_in_rad < 0: # Clockwise below x-axis; convert to an obtuse angle above x-axis.
                        angle_in_rad += (2 * math.pi)
                    angles_in_rad.append(angle_in_rad)
                    
                    # Average angle is computed with circular mean, i.e., mean of cos and sin then later atan2
                    # No need to divide both by flow_count since it would be cancelled in atan2 upon division
                    cos_sum += math.cos(angle_in_rad)
                    sin_sum += math.sin(angle_in_rad)
                    
                    flow_count += 1      
                    
        # End outer loop
        if flow_count == 0:
            return image
        angle_mean = circmean(angles_in_rad)
        
        avg_beg_x = round(beg_x_sum / flow_count, 2)
        avg_beg_y = round(beg_y_sum / flow_count, 2)
        avg_length = round(lengths_sum / flow_count, 2)
        
        csv_file = open(os.path.join(self.output_parent_dir, 'output-flows-averaging', "averages.csv"), "a")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([cell_order, contraction_id, bounding_frames, cell_y_beg, cell_x_beg, cell_y_end, cell_x_end, avg_beg_y, avg_beg_x, math.degrees(angle_mean), avg_length])
        
        x =  round(avg_beg_x + avg_length * math.cos(angle_mean), 2)
        y =  round(avg_beg_y + avg_length * math.sin(angle_mean), 2)
        
        image = cv2.arrowedLine(image, (int(avg_beg_x), int(avg_beg_y)), (int(x), int(y)), (0, 0, 0), thickness=3, tipLength = 0.4)
        
        if output_image_path: # to save the image only after the last grid cell instead of saving many times
            image = Image.fromarray(image)
            image.save(output_image_path)
            image = np.array(image) # because this function is expected to always return the image as a numpy array
            
        return image
        
        
    def summarize_flows_in_grid(self, contraction_file, contraction_time_div, flows_colors, contraction_beg_frame, primary_peak):
        contraction_id = contraction_file.split('/')[-1].split('.')[0]
        output_dir = os.path.join(self.output_parent_dir, 'output-flows-averaging', contraction_id)
        os.makedirs(output_dir, exist_ok=True)
        
        if self.grid_div > self.image_cols**0.5: # some random limit for the number of columns of the big grid, sqrt(self.image_cols)
            self.grid_div = 4
            
        df = pd.read_csv(contraction_file, header=None)
        df_rows, df_cols = df.shape
        
        if contraction_time_div > df_rows**0.5: # some random limit for the number of images/splits/frames
            contraction_time_div = 3
        contraction_splits = self.get_contraction_time_splits(contraction_time_div, df_rows, primary_peak)
        grid_rows, grid_cols = self.get_big_grid_cells()
        
        for frame in range(len(contraction_splits) - 1):
            frame_beg = contraction_splits[frame]
            frame_end = contraction_splits[frame + 1]
            
            frame_beg_order = contraction_beg_frame + frame_beg
            frame_end_order = contraction_beg_frame + frame_end - 1
            image_path = get_file_name_from_dir(self.input_dir, frame_end_order)
            
            image = Image.open(image_path)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cell_order = 1
            output_image_path = ''
            for row in range(len(grid_rows) - 1):
                for col in range(len(grid_cols) - 1):
                    cell_x_beg = grid_cols[col]
                    cell_x_end = grid_cols[col + 1]
                    cell_y_beg = grid_rows[row]
                    cell_y_end = grid_rows[row + 1]
                    
                    if (row == len(grid_rows) - 1 - 1) and (col == len(grid_cols) - 1 - 1):
                        output_image_path = contraction_id + '_[' + str(frame_beg_order) + '-' + str(frame_end_order) + '].png'
                        output_image_path = os.path.join(output_dir, output_image_path)
                    image = self.summarize_flows_in_grid_cell(image, df, flows_colors, frame_beg, frame_end, cell_x_beg, cell_x_end, cell_y_beg, cell_y_end, output_image_path, contraction_id, str(frame_beg) + "_" + str(frame_end), cell_order)
                    cell_order += 1
            self.draw_big_grid_cells(image, grid_rows, grid_cols, output_image_path)
            
            
