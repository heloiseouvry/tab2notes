import cv2
import numpy as np

def detect_intensity_along_axis(img, intensity, ax):
    mask = np.sum(img, axis=ax) >= intensity
    idx = []
    i0=None
    for (i,b) in enumerate(mask):
        if b == True:
            if i0 == None:
                i0 = i
            if i == mask.shape[0] - 1:
                idx.append((i0,i))
        elif b == False and i0 != None:
            idx.append((i0,i))
            i0=None
    return idx

def get_white_lines_extremum(img):
    line = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = np.median(line, axis=1) >= 250
    limits = np.zeros(mask.shape, dtype='uint8')
    limits[1:-1] = (np.bitwise_xor(mask[1:-1], mask[0:-2])) * 255
    idx = np.where(limits == 255)[0]
    idx = [(idx[i], idx[i+1]) for i in range(0,len(idx),2)]
    return idx

def staff_idx(img):
    return get_white_lines_extremum(img[:,0:10])

def col_idx(img):
    return get_white_lines_extremum(np.transpose(img))

def is_bold_staff_col(col_idx):
    for i in col_idx:
        if i[1]-i[0] > 7:
            return (i[0],i[1])

def is_last_part(img):
    bold_idx = is_bold_staff_col(col_idx(img))
    if bold_idx:
        if np.sum(img[:,bold_idx[1]+1:]) != 0: bold_idx = None
    return bold_idx

def get_extremum(idx):
    for (i,j) in enumerate(idx):
        if i == len(idx)-1: break
        print(f'{j}>{idx[i+1]}')

def remove_white_lines(img, idx):
    res = np.copy(img)
    for i in idx:
        for j in range(img.shape[1]):
            if sum(img[i[0]-5:i[0],j]) == 0 and sum(img[i[1]:i[1]+5,j]) == 0:
                res[i[0]:i[1],j] = 0
    return res

def remove_staff_idx(img, idx):
    return remove_white_lines(img, idx)

def remove_col_idx(img, idx):
    return np.transpose(remove_white_lines(np.transpose(img), idx))

def get_digit_idx(img):
    dgt_idx = []
    for i in detect_intensity_along_axis(img, 255, 0):
        if i[1]-i[0] == 1: continue
        roi = img[:,i[0]:i[1]]
        dgt = detect_intensity_along_axis(roi, 255, 1)
        for d in dgt:
            if d[1]-d[0] == 1: continue
            roi_2 = roi[d[0]:d[1],:]
            dgt2 = detect_intensity_along_axis(roi_2, 255, 0)
            for d2 in dgt2:
                if d2[1]-d2[0] == 1: continue
                dgt_idx.append([(d[0],i[0]+d2[0]),(d[1],i[0]+d2[1]+1)])
    return dgt_idx

def get_digit_img(img):
    dgt_img = []
    for d in get_digit_idx(img):
        dgt_img.append(img[d[0][0]:d[1][0],d[0][1]:d[1][1]])
    return dgt_img