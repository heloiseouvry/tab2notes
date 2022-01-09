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
        parts['original'] = preprocess.extract_parts(img)
        parts_0 = preprocess.invert_img(parts['original'][0])
        thresh_0 = preprocess.thresh_img(parts_0)
        
        staff_idx = detect.staff_idx(parts_0)
        col_idx = detect.col_idx(parts_0)

        removed = detect.remove_staff_idx(thresh_0, staff_idx)
        removed = detect.remove_col_idx(removed, col_idx)

        dgt_idx = detect.get_digit_idx(removed)
        dgt_img = detect.get_digit_img(removed)

        # cv2.imwrite(r'..\results\parts_0.jpg',parts_0)
        # cv2.imwrite(r'..\results\thresh_0.jpg',thresh_0)
        # cv2.imwrite(r'..\results\removed.jpg',removed)

        translated_part = copy.deepcopy(parts['original'][0])
        for (i,d) in enumerate(dgt_img):
            dgt = classif.with_digit_template(d)
            d_idx = dgt_idx[i]
            note = classif.to_note(dgt,d_idx,staff_idx,notation='fr')
            translated_part = postprocess.paste_note(translated_part,d_idx,note)
        translated_part = postprocess.bold_bottom_staff(translated_part, staff_idx)
        # cv2.imwrite(r'..\results\translated_part.jpg',translated_part)
        