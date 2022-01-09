import argparse
import copy
import classif
import detect
import preprocess
import postprocess
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help='path to input image file (required)')
parser.add_argument('-d','--dest', help='path to output image file')
parser.add_argument('-v','--verbose', help='more verbose')
args = parser.parse_args()

if __name__ == "__main__":

    # img = preprocess.read_image(args.input, 0)
    img = preprocess.read_image(r'..\data\arpege.jpg', 0)
    translated_img = copy.deepcopy(img)
    if preprocess.isGPformat(img):
        nb_parts = 5
        empty_array = [[] for i in range(nb_parts)]
        parts = {
            'original': copy.deepcopy(empty_array),
            'inv': copy.deepcopy(empty_array),
            'thresh': copy.deepcopy(empty_array),
            'wo_staff': copy.deepcopy(empty_array),
            'translated': copy.deepcopy(empty_array),
            'staff_line' : copy.deepcopy(empty_array),
            'staff_col' : copy.deepcopy(empty_array),
            'digits' : [{'idx': [],'img': [],'classif': [],'note':[]} for i in range(nb_parts)]  
        }
        parts['original'] = preprocess.extract_parts(img)[0]
        parts['idx'] = preprocess.extract_parts(img)[1]

        for p in range(nb_parts):
            parts['inv'][p] = preprocess.invert_img(parts['original'][p])
            parts['thresh'][p] = preprocess.thresh_img(parts['inv'][p])
        
            parts['staff_line'][p] = detect.staff_idx(parts['thresh'][p])
            parts['staff_col'][p] = detect.col_idx(parts['thresh'][p])

            parts['wo_staff'][p] = detect.remove_staff_idx(parts['thresh'][p], parts['staff_line'][p])
            parts['wo_staff'][p] = detect.remove_col_idx(parts['wo_staff'][p], parts['staff_col'][p])

            parts['digits'][p]['idx'] = detect.get_digit_idx(parts['wo_staff'][p])
            parts['digits'][p]['img'] = detect.get_digit_img(parts['wo_staff'][p])

            # cv2.imwrite(f'..\\results\\parts_{p}.jpg',parts['inv'][p])
            # cv2.imwrite(f'..\\results\\thresh_{p}.jpg',parts['thresh'][p])
            # cv2.imwrite(f'..\\results\\remove{p}.jpg',parts['wo_staff'][p])

            parts['translated'][p] = copy.deepcopy(parts['original'][p])

            for (i,d) in enumerate(parts['digits'][p]['img']):
                dgt = classif.with_digit_template(d)
                d_idx = parts['digits'][p]['idx'][i]
                note = classif.to_note(dgt,d_idx,parts['staff_line'][p],notation='fr')
                parts['translated'][p] = postprocess.paste_note(parts['translated'][p],d_idx,note)
            
            parts['translated'][p] = postprocess.bold_bottom_staff(parts['translated'][p], parts['staff_line'][p])
            # cv2.imwrite(f'..\\results\\parts_translated{p}.jpg',parts['translated'][p])
            
            # Pasting all back on the image
            translated_img[parts['idx'][p][0][0]:parts['idx'][p][1][0],parts['idx'][p][0][1]:parts['idx'][p][1][1]] = parts['translated'][p]
        cv2.imwrite(f'..\\results\\translated_img.jpg',translated_img)
