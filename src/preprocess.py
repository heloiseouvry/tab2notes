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