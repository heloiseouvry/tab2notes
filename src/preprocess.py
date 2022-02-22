import os.path
import cv2
import numpy as np
from pdf2image import convert_from_path
from urllib.request import urlopen
from . import detect


def path_split(path):
    filename = path.split(os.path.sep)[-1].split(".")[0]
    format = path.split(os.path.sep)[-1].split(".")[1]
    start = path[:-(len(filename)+len(format)+1)]
    return [start, filename, format]

def isurl(input):
    return True if input.startswith('http') else False

def ispdf(input):
    return True if input.split('.')[-1] == 'pdf' else False

def pdf2jpg(filepath):
    """Saves JPG image(s) from PDF file.
    If there is more than one page, it will number the filename.
    Args:
        filepath (str): path to the PDF file
    """
    if not isinstance(filepath,str):
        raise ValueError('Filepath should be a string')
    if not ispdf(filepath):
        raise ValueError('File is not a pdf')
    try:
        if isurl(filepath):
            file = convert_from_path(bytearray(urlopen(filepath).read()), fmt='jpg')
        else :
            if not os.path.isfile(filepath):
                raise ValueError('File does not exist')
            file = convert_from_path(filepath)
        [start, filename, _] = path_split(filepath)
        filenum = ''
        for (i,f) in enumerate(file):
            if i >= 1:
                filenum = f'_{i+1}'
            save_path = f'{start}{filename}{filenum}.jpg'
            print(f'save_path : {save_path} | i = {i}')
            f.save(save_path, 'jpeg')
    except Exception as e:
        print(f"Issue in PDF conversion: {e}")
    return i+1

def get_no_pages(input):
    if not isinstance(input,str):
        raise ValueError('Filepath should be a string')
    if not os.path.isfile(input):
        raise ValueError('Filepath is not valid')
    return pdf2jpg(input) if ispdf(input) else 1

def read_image(filepath, color=True):
    """Read an image with OpenCV and return it as a numpy array
    Args:
        filepath (str): path to the image
        color (bool): read image in color mode (True) or grayscale mode (False)

    Returns:
        image as a numpy array
    """

    if not isinstance(filepath,str):
        raise ValueError('Filepath should be a string')
    if not os.path.isfile(filepath):
        raise ValueError(f'Filepath is not valid [{filepath}]')
    return cv2.imread(filepath, int(color))

def isGPformat(img):
    h,w = img.shape
    if round(h/w,1) != 1.4:
        print('Input image should be in a GuitarPro format standard')
        return False
    return True

def extract_parts(GPimg):
    h,w = GPimg.shape
    part_height = int(0.062*h)
    part_width = int(0.84*w)
    start_cols = int(0.107*w)
    parts = []
    parts_idx = []
    idx = detect.detect_intensity_along_axis(invert_img(GPimg), 255, 1)
    for id in idx:
        detected_height = id[1]-id[0]
        if detected_height / h > 0.10:
            parts.append(GPimg[id[1]-part_height:id[1],start_cols:start_cols+part_width])
            parts_idx.append([(id[1]-part_height,start_cols),(id[1],start_cols+part_width)])
    return [parts,parts_idx]

def invert_img(img):
    return cv2.bitwise_not(img)

def thresh_img(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]