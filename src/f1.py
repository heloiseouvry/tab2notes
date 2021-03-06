import argparse
import copy
from . import classif, detect, preprocess, postprocess
import numpy as np
import cv2

# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--input', help='path to input image file (required)')
# parser.add_argument('-d','--dest', help='path to output image file')
# parser.add_argument('-v','--verbose', help='more verbose', action="store_true")
# args = parser.parse_args()

def translate(input, output, verbose=False):
    img = preprocess.read_image(input, 0)
    if chr(47) in input:
        input_name = input.split(chr(47))[-1].split(".")[0]
    if chr(92) in input:
        input_name = input.split(chr(92))[-1].split(".")[0]
    # img = preprocess.read_image(r'..\data\arpege.jpg', 0)
    translated_img = copy.deepcopy(img)
    if preprocess.isGPformat(img):
        parts = {}
        parts['original'], parts['idx'] = preprocess.extract_parts(img)
        nb_parts = len(parts['original'])
        if verbose:
            print(f'nb_parts = {nb_parts}')
        empty_array = [[] for _ in range(nb_parts)]
        parts['inv'] = copy.deepcopy(empty_array)
        parts['thresh'] = copy.deepcopy(empty_array)
        parts['wo_staff'] = copy.deepcopy(empty_array)
        parts['translated'] = copy.deepcopy(empty_array)
        parts['staff_line'] = copy.deepcopy(empty_array)
        parts['staff_col'] = copy.deepcopy(empty_array)
        parts['digits'] = [{'idx': [],'img': [],'classif': [],'note':[]} for _ in range(nb_parts)]  

        for p in range(nb_parts):
            parts['inv'][p] = preprocess.invert_img(parts['original'][p])
            parts['thresh'][p] = preprocess.thresh_img(parts['inv'][p])
        
            parts['staff_line'][p] = detect.staff_idx(parts['thresh'][p])
            parts['staff_col'][p] = detect.col_idx(parts['thresh'][p])
            bold = detect.is_bold_staff_col(parts['staff_col'][p])
            last = detect.is_last_part(parts['thresh'][p])

            if last:
                parts['original'][p] = parts['original'][p][:,:last[1]]
                parts['inv'][p] = parts['inv'][p][:,:last[1]]
                parts['thresh'][p] = parts['thresh'][p][:,:last[1]]
                parts['idx'][p][1] = (parts['idx'][p][1][0], parts['idx'][p][0][1] + last[1])

            parts['wo_staff'][p] = detect.remove_staff_idx(parts['thresh'][p], parts['staff_line'][p])
            parts['wo_staff'][p] = detect.remove_col_idx(parts['wo_staff'][p], parts['staff_col'][p])

            # cv2.imwrite(f'..\\results\\parts_{p}.jpg',parts['inv'][p])
            # cv2.imwrite(f'..\\results\\thresh_{p}.jpg',parts['thresh'][p])
            # cv2.imwrite(f'..\\results\\remove{p}.jpg',parts['wo_staff'][p])

            parts['digits'][p]['idx'] = detect.get_digit_idx(parts['wo_staff'][p])
            parts['digits'][p]['img'] = detect.get_digit_img(parts['wo_staff'][p])

            parts['translated'][p] = copy.deepcopy(parts['original'][p])

            parts['translated'][p] = postprocess.bold_bottom_staff(parts['translated'][p], parts['staff_line'][p])
            for (i,d) in enumerate(parts['digits'][p]['img']):
                dgt_ratio = d.shape[1]/d.shape[0]
                if dgt_ratio > 0.8:
                    continue
                dgt = classif.with_digit_template(d)
                d_idx = parts['digits'][p]['idx'][i]
                note = classif.to_note(dgt,d_idx,parts['staff_line'][p],notation='fr')
                parts['translated'][p] = postprocess.paste_note(parts['translated'][p],d_idx,note)
            
            # cv2.imwrite(f'..\\results\\parts_translated{p}.jpg',parts['translated'][p])
            
            # Pasting all back on the image
            translated_img[parts['idx'][p][0][0]:parts['idx'][p][1][0],parts['idx'][p][0][1]:parts['idx'][p][1][1]] = parts['translated'][p]
        if output:
            cv2.imwrite(output + input_name + '_translated.jpg', translated_img)
        return input_name + '_translated.jpg'

# if __name__ == "__main__":
#     translate(args.input, args.output, args.verbose)