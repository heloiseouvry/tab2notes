import cv2
import numpy as np

def bold_bottom_staff(img, staff_idx):
    """Bolding the bottom staff line
    
    Args:
        img (array): array of the image
        staff_idx (list of tuples): tuples of the staff indices

    Returns:
        modified image
    """
    res = np.copy(img)
    res[staff_idx[-1][0]-2 : staff_idx[-1][1]+2 , :] = 0
    return res