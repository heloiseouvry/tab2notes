import os.path
import cv2
from pdf2image import convert_from_path
import detect

def pdf2jpg(filepath):
    """Saves JPG image(s) from PDF file.
    If there is more than one page, it will number the filename.
    Args:
        filepath (str): path to the PDF file
    """
    if not isinstance(filepath,str):
        raise ValueError('Filepath should be a string')
    if not os.path.isfile(filepath):
        raise ValueError('File does not exist')
    if not filepath.split('.')[-1] == 'pdf':
        raise ValueError('File is not a pdf')
    try:
        file = convert_from_path(filepath)
        filepath = filepath[:-4]
        filenum = ''
        for (i,f) in enumerate(file):
            if i >= 1:
                filenum = f'_{i+1}'
            f.save(f'{filepath}{filenum}.jpg', 'jpeg')
    except:
        print("Issue in PDF conversion")
    

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
        raise ValueError('Filepath is not valid')

    if filepath.split('.')[-1] == 'pdf':
        pdf2jpg(filepath)
        filepath = f'{filepath[:-4]}.jpg'

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