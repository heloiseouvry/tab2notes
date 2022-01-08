import argparse
import copy
import preprocess

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help='path to input image file (required)')
parser.add_argument('-d','--dest', help='path to output image file')
parser.add_argument('-v','--verbose', help='more verbose')
args = parser.parse_args()

if __name__ == "__main__":

    nb_parts = 4
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

    img = preprocess.read_image(args.input, 0)
    if preprocess.isGPformat(img):
        parts = preprocess.extract_parts(img)
        print(parts[0].shape)