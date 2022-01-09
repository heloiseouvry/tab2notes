import cv2
import numpy as np

def detect_intensity_along_axis(img, intensity, ax):
    mask = np.sum(img, axis=ax) >= intensity
    limits = np.zeros(mask.shape, dtype='uint8')
    limits[1:-1] = (np.bitwise_xor(mask[1:-1], mask[0:-2])) * intensity
    idx = np.where(limits == intensity)[0]
    idx = [(idx[i], idx[i+1]) for i in range(0,len(idx),2)]
    return idx

def get_white_lines(img):
    line = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    line = np.median(line, axis=1)
    return np.where(line >= 250)[0]

def staff_idx(img):
    return get_white_lines(img[:,0])

def col_idx(img):
    return get_white_lines(np.transpose(img))

def remove_white_lines(img, idx):
    res = np.copy(img)
    for i in idx:
        for j in range(img.shape[1]):
            if sum(img[i-5:i,j]) == 0 and sum(img[i+1:i+5,j]) == 0:
                res[i,j] = 0
    return res