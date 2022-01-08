import os.path
import cv2
from pdf2image import convert_from_path

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
    mesure_height = int(0.053*h)
    mesure_width = int(0.84*w)
    start_rows = [int(r * h) for r in [0.223, 0.363, 0.5, 0.63]]
    # start_rows = [int(r * h) for r in [0.223, 0.363, 0.5, 0.63, 0.756]]
    start_cols = int(0.107 * w)
    nb_parts = 4
    # nb_parts = 5
    parts = [[] for i in range(nb_parts)]
    for (i,r) in enumerate(start_rows):
        parts[i] = GPimg[r:r+mesure_height,start_cols:start_cols+mesure_width]
    return parts