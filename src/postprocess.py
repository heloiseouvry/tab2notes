import copy
import cv2
import numpy as np
import classif

def bold_bottom_staff(img, staff_idx):
    """Bolding the bottom staff line
    
    Args:
        img (array): array of the image
        staff_idx (list of tuples): tuples of the staff indices

    Returns:
        modified image (copy)
    """
    res = np.copy(img)
    res[staff_idx[-1][0]-2 : staff_idx[-1][1]+2 , :] = 0
    return res

def paste_note(base_img, dgt_idx, dgt_note):
    """Paste the note image at corresponding index on an image 
    
    Args:
        base_img (array): array of the base image
        dgt_idx (tuple): tuple of the digit indices
        dgt_note (str): name of the note

    Returns:
        modified image (copy)
    """
    out_img = copy.deepcopy(base_img)
    # Erasing the old digit number
    out_img[dgt_idx[0][0]:dgt_idx[1][0],dgt_idx[0][1]:dgt_idx[1][1]] = 255
    # Getting the corresponding note image
    note_img = classif.notes_fr_img[dgt_note]
    # Resizing
    dgt_h = dgt_idx[1][0] - dgt_idx[0][0]
    note_img = cv2.resize(note_img, (int(note_img.shape[1]*(dgt_h/note_img.shape[0])),dgt_h))
    # Centering
    centroid = (dgt_idx[0][0]+int((dgt_idx[1][0]-dgt_idx[0][0])/2),int(dgt_idx[0][1]+(dgt_idx[1][1]-dgt_idx[0][1])/2))
    start_row = centroid[0] - int(note_img.shape[0] / 2)
    start_col = centroid[1] - int(note_img.shape[1] / 2)
    # Pasting
    out_img[start_row:start_row+note_img.shape[0],start_col:start_col+note_img.shape[1]] = note_img
    return out_img