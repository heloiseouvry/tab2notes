import cv2
import numpy as np

digit_template_img = []
for i in range(10):
    digit_template_img.append(cv2.threshold(cv2.imread(f'..\\data\\digit_template\\{i}.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
digit_template_h = digit_template_img[0].shape[0]
digit_template_w = digit_template_img[0].shape[1]

def with_digit_template(dgt_img):
    """ Classify an image of a digit according to some digit template images (0-9).
    It takes the MSE of images difference.

    Args:
        dgt_img (array): array of the image

    Returns:
        integer corresponding to the digit
    """
    classif_err = []
    height = dgt_img.shape[0]
    width = dgt_img.shape[1]
    for i in range(10):
        if height != digit_template_h or width != digit_template_w:
            dgt_img = cv2.resize(dgt_img, (digit_template_w,digit_template_h), interpolation=cv2.INTER_NEAREST)        
            
        sub = np.bitwise_xor(dgt_img.astype(bool), digit_template_img[i].astype(bool))
        classif_err.append(int(100*(np.mean(np.square(sub)))))
    #     print(f'Taux d\'erreur pour {i} : {classif_err[i]} %')
    # print(f'--> Classification = {np.argmin(classif_err)}')
    return np.argmin(classif_err)