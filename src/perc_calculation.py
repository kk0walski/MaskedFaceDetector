import cv2 
import numpy as np
from PIL import Image

PERC_THR = 0.8

def calculate_masking_percentage(image, skin_mask, restr_mask, haar_regions):
    output = np.array(image)
    skin_m = cv2.cvtColor(np.array(skin_mask), cv2.COLOR_RGB2GRAY)
    restr_m = cv2.cvtColor(np.array(restr_mask), cv2.COLOR_RGB2GRAY)
    
    # filter skim mask by haar cascade detected areas
    for _, regions in haar_regions.items():
        for (xmin, ymin), (xmax, ymax) in regions:
            skin_part = skin_m[ymin:ymax, xmin:xmax]
            u_px, cnt_px = np.unique(skin_part, return_counts=True)
            if 255 in u_px:
                id_255 = np.where(u_px == 255)[0]
                s_h, s_w = skin_part.shape
                if cnt_px[id_255] >= PERC_THR*s_h*s_w:
                    # make certain area white
                    skin_m[ymin:ymax, xmin:xmax] = np.full(skin_part.shape, 255)
            
    # combine skin and restriction mask (bitwise and)
    final_m = skin_m & restr_m
    # calculate percentage
    u_px_restr, cnt_px_restr = np.unique(restr_m, return_counts=True)
    u_px_final, cnt_px_final = np.unique(final_m, return_counts=True)
    perc = 1. - cnt_px_final[np.where(u_px_final == 255)[0]] / cnt_px_restr[np.where(u_px_restr == 255)[0]]
    
    output = cv2.bitwise_and(output, output, mask = final_m)
    
    final_m = cv2.cvtColor(final_m, cv2.COLOR_GRAY2RGB)
    
    final_m = Image.fromarray(final_m)
    output = Image.fromarray(output)
    return perc[0], final_m, output
    
def draw_roi(image, jawline_pts):
    output = np.array(image)
    
    jawline_pts = sorted(jawline_pts, key=lambda k: k[0])
    top_left_corner = (jawline_pts[0][0], 0)
    top_right_corner = (jawline_pts[-1][0], 0)
    
    face_roi_pts = list([top_left_corner, *jawline_pts, top_right_corner])
    
    for i in range(len(face_roi_pts)):
        start = face_roi_pts[i]
        end = face_roi_pts[(i+1)%len(face_roi_pts)]
        cv2.line(output, start, end, (0,255,0), 2)
    
    output = Image.fromarray(output)
    
    return output

