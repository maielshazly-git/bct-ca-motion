import os
import cv2
import csv
import glob
import numpy as np
import pandas as pd
from PIL import Image
from alive_progress import alive_bar
from utility import get_file_name_from_dir, get_divisors, get_bct_indices


class MotionTracker:
    def __init__(self, input_dir, output_parent_dir, masks_dir, left_ips, right_ips):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.masks_dir = masks_dir
        self.frames_output_dir = ''
        self.flows_output_dir = ''
        self.left_ips = left_ips
        self.right_ips = right_ips
        
    def get_frames_output_dir(self):
        return self.frames_output_dir
        
        
    def get_flows_output_dir(self):
        return self.flows_output_dir
        
        
    def lucas_kanade_method(self, time_bet_frame_present_msec=1):               
        contraction_id, save_flows, features_ids = 0, False, []
        cap = cv2.VideoCapture(get_file_name_from_dir(self.input_dir))
        
        beg_frames_to_skip = 10 # I wait for the algorithm to stabilize, due to initial jumps
        if self.left_ips is None: # for non-contracting drugs
            self.left_ips = [beg_frames_to_skip + 1]
        elif beg_frames_to_skip > self.left_ips[0]: # for contracting drugs that have very early first contraction
            self.left_ips[0] = beg_frames_to_skip + 1
            if len(self.left_ips) > 1:
                if self.left_ips[0] >= self.left_ips[1] or self.left_ips[0] >= self.right_ips[0]:
                    self.left_ips.pop(0)
                    self.right_ips.pop(0)
                    
        for i in range(beg_frames_to_skip):
            ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)
        
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        i = 10
        rows, cols = old_gray.shape
        row_div, col_div = get_divisors(rows)[i], get_divisors(cols)[i]
        
        frame_order = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        bct_indices = get_bct_indices(get_file_name_from_dir(self.masks_dir, frame_order))
        
        self.frames_output_dir = os.path.join(self.output_parent_dir, 'output-frames')
        os.makedirs(self.frames_output_dir, exist_ok=True)
            
        self.flows_output_dir = os.path.join(self.output_parent_dir, 'output-flows')
        os.makedirs(self.flows_output_dir, exist_ok=True)
        
        i, features_ori_count = 0, 0
        p0_orig = []
        while i < len(bct_indices):
            p0_orig.append(bct_indices[i])
            features_ori_count += 1
            if i % cols == 0:
                i += (row_div * cols)
            i += col_div
            
        p0 = np.array(p0_orig, dtype='float32')
        color = np.random.randint(0, 150, (len(p0), 3))
        
        with alive_bar(int(self.right_ips[-1] - beg_frames_to_skip)) as bar:
            print('Press Esc to stop at the current contraction and move on with these contractions only.')
            while True:
                ret, frame = cap.read()
                if not ret or contraction_id == len(self.left_ips):
                    break
                    
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_order = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                if frame_order == self.left_ips[contraction_id]:
                    save_flows = True
                    p0 = np.array(p0_orig, dtype='float32')
                    mask = np.zeros_like(old_frame)
                    features_ids = np.arange(len(p0))
                    
                    csv_file = open(os.path.join(self.flows_output_dir, "contraction_" + str(contraction_id + 1) + ".csv"), "a")
                    csv_writer = csv.writer(csv_file)
                    csv_row = []
                    
                    for i in range(len(p0)):
                        feature = p0[i][0] # due to the format of p0 [ [[col, row]], [[col, row]], [[col, row]] ]
                        csv_row.append(str(int(feature[0])) + '_' + str(int(feature[1])))
                    csv_writer.writerow(csv_row)
                    
                elif frame_order == self.right_ips[contraction_id]:
                    save_flows = False
                    p0 = np.array(p0_orig, dtype='float32')
                    mask = np.zeros_like(old_frame)
                    contraction_id += 1
                    features_ids = np.arange(len(p0))
                    
                #print('frame_order:{}	save_flows:{}'.format(frame_order, save_flows))
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                try:
                    # This line avoids errors in case no st is found, then good_new and good_old aren't defined.
                    good_old, good_new = np.array([[]], dtype='float32'), np.array([[]], dtype='float32')
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    csv_row = []
                    x = 0 # index for features_ids, good_old and good_new
                    st = st.flatten()
                    
                    if len(st) > 0 and len(features_ids) > 0:
                        temp = []
                        for i in range(len(st)):
                            if st[i] == 1:
                                temp.append(features_ids[i])
                        features_ids = temp
                        
                    for i in range(features_ori_count):
                        if save_flows and i != features_ids[x]:
                            csv_row.append(None)
                            
                        else:
                            a, b = good_new[x].ravel().astype(int) # a, b are col, row, respectively
                            c, d = good_old[x].ravel().astype(int)
                            csv_row.append(str(a) + '_' + str(b))
                            
                            # Draw the tracks
                            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 4)
                            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                            x += 1
                            if x == len(good_new):
                                break
                            
                    if save_flows:
                        csv_writer.writerow(csv_row)
                        
                except Exception as e:
                    pass
                    #if st is None:
                        #print("No more paths found in the current contraction ...")
                    #else:
                        #print("Crashed ...", str(e))
                        
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                
                img = cv2.add(frame, mask)
                cv2.imshow("frame", img)
                cv2.imwrite(os.path.join(self.frames_output_dir, str(frame_order) + '.png'), img)
                
                if cv2.waitKey(time_bet_frame_present_msec) == 27: # 27 is esc on the keyboard
                    break
                    
                bar()
        return contraction_id, cols, color
        
        
