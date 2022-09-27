import argparse
from cmath import log
import copy
from . import classif, detect, preprocess, postprocess
import numpy as np
import cv2

# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--input', help='path to input image file (required)')
# parser.add_argument('-d','--dest', help='path to output image file')
# args = parser.parse_args()

def translate_img(input, logger=None):
    img = preprocess.read_image(input, 0)
    translated_img = copy.deepcopy(img)
    if preprocess.isGPformat(img):
        parts = {}
        parts['original'], parts['idx'] = preprocess.extract_parts(img)
        nb_parts = len(parts['original'])
        empty_array = [[] for _ in range(nb_parts)]
        parts['inv'] = copy.deepcopy(empty_array)
        parts['thresh'] = copy.deepcopy(empty_array)
        parts['wo_staff'] = copy.deepcopy(empty_array)
        parts['translated'] = copy.deepcopy(empty_array)
        parts['staff_line'] = copy.deepcopy(empty_array)
        parts['staff_col'] = copy.deepcopy(empty_array)
        parts['digits'] = [{'idx': [],'img': [],'classif': [],'note':[]} for _ in range(nb_parts)]  

        if logger:
            logger.info(f"Nombre de parties : {nb_parts}")

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

            if logger:
                logger.info(f"... Parts {p}/{nb_parts} ...")

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
        return translated_img
        
def translate(input, output, logger=None):
    no_pages = preprocess.get_no_pages(input)
    [start, input_name, format] = preprocess.path_split(input)
    print(f'input = {input}')
    print(f'output = {output}')
    translation = []
    input_img_path = f'{start}{input_name}.jpg'
    output_img_path = f'{input_name}_translated.jpg'
    if logger:
        logger.info(f"""
        {' PARTITION ':#^50}
        Nom de la partition : {input_name}
        Nombre de pages : {no_pages}
        {' Chemins ':-^40}
        Input : {input_img_path}
        Output : {output_img_path}
        {' TRADUCTION ':#^50}
        """)
    for i in range(no_pages):
        if logger:
            logger.info(f"------- Page {i}/{no_pages} -------")
        if i >= 1:
            input_img_path = f'{start}{input_name}_{i+1}.jpg'
            output_img_path = f'{input_name}_{i+1}_translated.jpg'
        print(f'input_img_path = {input_img_path}')
        print(f'output_img_path = {output_img_path}')
        translated_img = translate_img(input_img_path, logger=logger)
        translation.append(output_img_path)
        print(f'translation = {translation}')
        if output:
            cv2.imwrite(start+output_img_path, translated_img)
    return translation

# if __name__ == "__main__":
#     translate(args.input, args.output)