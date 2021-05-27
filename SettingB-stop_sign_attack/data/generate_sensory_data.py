import numpy as np
import cv2
import pickle


def foreground_segmentation(img):
    mask = np.zeros((128,128),np.uint8)
    ## Foreground Segmentation
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,126,126)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,15,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    return mask2

def preprocess(img):

    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    temp[:,:,1] = cv2.equalizeHist(temp[:,:,1])
    temp[:,:,2] = cv2.equalizeHist(temp[:,:,2])
    temp = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    temp = cv2.GaussianBlur(temp,(7,7),0)
    mask = foreground_segmentation(temp)

    mask_larger = cv2.resize(mask,(138,138))
    mask_smaller = cv2.resize(mask,(90,90))

    border = img.copy()
    for i in range(3):
        border[:,:,i] = border[:,:,i] * mask_larger[5:133,5:133]
    mask[:,:] = 1
    mask[19:109,19:109] = 1 - mask_smaller[:,:]
    for i in range(3):
        border[:,:,i] = border[:,:,i] * mask

    content = img.copy()
    content = cv2.cvtColor(content,cv2.COLOR_BGR2GRAY)
    content = cv2.GaussianBlur(content,(7,7),0)
    content = cv2.equalizeHist(content)
    content = cv2.Canny(content, 20, 50)
    mask[:,:] = 0
    mask[19:109,19:109] = mask_smaller[:,:]
    content *= mask

    return border, content



for sample_path in ['stop_sign_adv_X_test.npy', 'stop_sign_X_test.npy']:

    X = np.load('./data/%s' % sample_path)

    num = len(X)

    border = [] 
    content = []

    for i in range(num):
        img = X[i]
        img = cv2.resize(img,(128,128))
        b_img, c_img = preprocess(img)
        border.append(b_img)
        content.append(c_img)

        print('[%s] %d/%d' % (sample_path,i+1,num))

    border = np.array(border)
    content = np.array(content)

    if sample_path == 'stop_sign_X_test.npy':        
        np.save('./data/stop_sign_border_test.npy',border)
        np.save('./data/stop_sign_content_test.npy',content)
    else:
        np.save('./data/stop_sign_adv_border_test.npy', border)
        np.save('./data/stop_sign_adv_content_test.npy', content)
