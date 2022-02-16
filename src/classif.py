import cv2
import numpy as np
import os

digit_template_img = []
for i in range(10):
    digit_template_img.append(cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\digit_template\\{i}.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
digit_template_h = digit_template_img[0].shape[0]
digit_template_w = digit_template_img[0].shape[1]


notes = {
    0: {0:'E', 1:'F', 2:'F#', 3:'G', 4:'G#', 5:'A', 6:'Bb', 7:'B', 8:'C', 9:'C#', 10:'D', 11:'Eb'},
    1: {0:'B', 1:'C', 2:'C#', 3:'D', 4:'Eb', 5:'E', 6:'F', 7:'F#', 8:'G', 9:'G#', 10:'A', 11:'Bb'},
    2: {0:'G', 1:'G#', 2:'A', 3:'Bb', 4:'B', 5:'C', 6:'C#', 7:'D', 8:'Eb', 9:'E', 10:'F', 11:'F#'},
    3: {0:'D', 1:'Eb', 2:'E', 3:'F', 4:'F#', 5:'G', 6:'G#', 7:'A', 8:'Bb', 9:'B', 10:'C', 11:'C#'},
    4: {0:'A', 1:'Bb', 2:'B', 3:'C', 4:'C#', 5:'D', 6:'Eb', 7:'E', 8:'F', 9:'F#', 10:'G', 11:'G#'},
    5: {0:'E', 1:'F', 2:'F#', 3:'G', 4:'G#', 5:'A', 6:'Bb', 7:'B', 8:'C', 9:'C#', 10:'D', 11:'Eb'}     
}

notes_fr = {
    0: {0:'Mi', 1:'Fa', 2:'Fa#', 3:'Sol', 4:'Sol#', 5:'La', 6:'Sib', 7:'Si', 8:'Do', 9:'Do#', 10:'Ré', 11:'Mib'},
    1: {0:'Si', 1:'Do', 2:'Do#', 3:'Ré', 4:'Mib', 5:'Mi', 6:'Fa', 7:'Fa#', 8:'Sol', 9:'Sol#', 10:'La', 11:'Sib'},
    2: {0:'Sol', 1:'Sol#', 2:'La', 3:'Sib', 4:'Si', 5:'Do', 6:'Do#', 7:'Ré', 8:'Mib', 9:'Mi', 10:'Fa', 11:'Fa#'},
    3: {0:'Ré', 1:'Mib', 2:'Mi', 3:'Fa', 4:'Fa#', 5:'Sol', 6:'Sol#', 7:'La', 8:'Sib', 9:'Si', 10:'Do', 11:'Do#'},
    4: {0:'La', 1:'Sib', 2:'Si', 3:'Do', 4:'Do#', 5:'Ré', 6:'Mib', 7:'Mi', 8:'Fa', 9:'Fa#', 10:'Sol', 11:'Sol#'},
    5: {0:'Mi', 1:'Fa', 2:'Fa#', 3:'Sol', 4:'Sol#', 5:'La', 6:'Sib', 7:'Si', 8:'Do', 9:'Do#', 10:'Ré', 11:'Mib'}     
}

notes_fr_img = {
    'Do': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\0.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'Ré': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\1.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'Mi': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\2.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'Fa': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\3.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'Sol': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\4.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'La': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\5.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'Si': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\6.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    'b': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\7.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    '#': cv2.threshold(cv2.imread(f'{os.path.dirname(__file__)}\\..\\data\\notes_fr\\8.jpg',0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
}
notes_fr_img['Do#'] = np.concatenate((notes_fr_img['Do'],notes_fr_img['#']), axis=1)
notes_fr_img['Mib'] = np.concatenate((notes_fr_img['Mi'],notes_fr_img['b']), axis=1)
notes_fr_img['Fa#'] = np.concatenate((notes_fr_img['Fa'],notes_fr_img['#']), axis=1)
notes_fr_img['Sol#'] = np.concatenate((notes_fr_img['Sol'],notes_fr_img['#']), axis=1)
notes_fr_img['Sib'] = np.concatenate((notes_fr_img['Si'],notes_fr_img['b']), axis=1)


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
        template = digit_template_img[i]
        if height != digit_template_img[i].shape[0] or width != digit_template_img[i].shape[1]:
            # dgt_img = cv2.resize(dgt_img, (digit_template_w,digit_template_h), interpolation=cv2.INTER_NEAREST)   # Resize digit to template size   
            template = cv2.resize(template, (width,height), interpolation=cv2.INTER_NEAREST)                                # Resize template to digit size     
        sub = np.bitwise_xor(dgt_img.astype(bool), template.astype(bool))
        classif_err.append(int(100*(np.mean(np.square(sub)))))
    #     print(f'Taux d\'erreur pour {i} : {classif_err[i]} %')
    # print(f'--> Classification = {np.argmin(classif_err)}')
    return np.argmin(classif_err)

def to_staff(dgt_idx, staff_idx):
    for (s_idx,s) in enumerate(staff_idx):
        if s[0] > dgt_idx[0][0] and s[1] < dgt_idx[1][0] :
            return s_idx

def to_note(dgt, dgt_idx, staff_idx, notation=''):
    if notation == 'fr':
        # print(f'digit = {dgt} on staff {to_staff(dgt_idx, staff_idx)}')
        return notes_fr[to_staff(dgt_idx, staff_idx)][dgt]
    else:
        return notes[to_staff(dgt_idx, staff_idx)][dgt]