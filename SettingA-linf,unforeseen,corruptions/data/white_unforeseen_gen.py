import numpy as np
import cv2


def foreground_segmentation(img):
    mask = np.zeros((128,128),np.uint8)
    ## Foreground Segmentation
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (1,1,126,126)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
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

natural_attack_mapping = {

    ## Fog
    '[clean]fog_256_adv_X_test.npy' : '[clean]fog_256_adv', 
    '[clean]fog_512_adv_X_test.npy' : '[clean]fog_512_adv',

    '[adv_4]fog_256_adv_X_test.npy' : '[adv_4]fog_256_adv', 
    '[adv_4]fog_512_adv_X_test.npy' : '[adv_4]fog_512_adv',

    '[adv_8]fog_256_adv_X_test.npy' : '[adv_8]fog_256_adv', 
    '[adv_8]fog_512_adv_X_test.npy' : '[adv_8]fog_512_adv',

    '[adv_16]fog_256_adv_X_test.npy' : '[adv_16]fog_256_adv', 
    '[adv_16]fog_512_adv_X_test.npy' : '[adv_16]fog_512_adv',

    '[adv_32]fog_256_adv_X_test.npy' : '[adv_32]fog_256_adv', 
    '[adv_32]fog_512_adv_X_test.npy' : '[adv_32]fog_512_adv',

    '[doa_5]fog_256_adv_X_test.npy' : '[doa_5]fog_256_adv', 
    '[doa_5]fog_512_adv_X_test.npy' : '[doa_5]fog_512_adv',

    '[doa_7]fog_256_adv_X_test.npy' : '[doa_7]fog_256_adv', 
    '[doa_7]fog_512_adv_X_test.npy' : '[doa_7]fog_512_adv',


    ## Snow
    '[clean]snow_0.25_adv_X_test.npy' : '[clean]snow_0.25_adv', 
    '[clean]snow_0.75_adv_X_test.npy' : '[clean]snow_0.75_adv', 

    '[adv_4]snow_0.25_adv_X_test.npy' : '[adv_4]snow_0.25_adv', 
    '[adv_4]snow_0.75_adv_X_test.npy' : '[adv_4]snow_0.75_adv', 

    '[adv_8]snow_0.25_adv_X_test.npy' : '[adv_8]snow_0.25_adv', 
    '[adv_8]snow_0.75_adv_X_test.npy' : '[adv_8]snow_0.75_adv', 

    '[adv_16]snow_0.25_adv_X_test.npy' : '[adv_16]snow_0.25_adv', 
    '[adv_16]snow_0.75_adv_X_test.npy' : '[adv_16]snow_0.75_adv', 

    '[adv_32]snow_0.25_adv_X_test.npy' : '[adv_32]snow_0.25_adv', 
    '[adv_32]snow_0.75_adv_X_test.npy' : '[adv_32]snow_0.75_adv', 

    '[doa_5]snow_0.25_adv_X_test.npy' : '[doa_5]snow_0.25_adv', 
    '[doa_5]snow_0.75_adv_X_test.npy' : '[doa_5]snow_0.75_adv', 

    '[doa_7]snow_0.25_adv_X_test.npy' : '[doa_7]snow_0.25_adv', 
    '[doa_7]snow_0.75_adv_X_test.npy' : '[doa_7]snow_0.75_adv', 
    

    ## JPEG
    '[clean]jpeg_0.125_adv_X_test.npy' : '[clean]jpeg_0.125_adv', 
    '[clean]jpeg_0.25_adv_X_test.npy' : '[clean]jpeg_0.25_adv', 

    '[adv_4]jpeg_0.125_adv_X_test.npy' : '[adv_4]jpeg_0.125_adv', 
    '[adv_4]jpeg_0.25_adv_X_test.npy' : '[adv_4]jpeg_0.25_adv', 

    '[adv_8]jpeg_0.125_adv_X_test.npy' : '[adv_8]jpeg_0.125_adv', 
    '[adv_8]jpeg_0.25_adv_X_test.npy' : '[adv_8]jpeg_0.25_adv', 

    '[adv_16]jpeg_0.125_adv_X_test.npy' : '[adv_16]jpeg_0.125_adv', 
    '[adv_16]jpeg_0.25_adv_X_test.npy' : '[adv_16]jpeg_0.25_adv', 

    '[adv_32]jpeg_0.125_adv_X_test.npy' : '[adv_32]jpeg_0.125_adv', 
    '[adv_32]jpeg_0.25_adv_X_test.npy' : '[adv_32]jpeg_0.25_adv', 

    '[doa_5]jpeg_0.125_adv_X_test.npy' : '[doa_5]jpeg_0.125_adv', 
    '[doa_5]jpeg_0.25_adv_X_test.npy' : '[doa_5]jpeg_0.25_adv', 

    '[doa_7]jpeg_0.125_adv_X_test.npy' : '[doa_7]jpeg_0.125_adv', 
    '[doa_7]jpeg_0.25_adv_X_test.npy' : '[doa_7]jpeg_0.25_adv', 


    ## Gabor
    '[clean]gabor_20_adv_X_test.npy' : '[clean]gabor_20_adv', 
    '[clean]gabor_40_adv_X_test.npy' : '[clean]gabor_40_adv',

    '[adv_4]gabor_20_adv_X_test.npy' : '[adv_4]gabor_20_adv', 
    '[adv_4]gabor_40_adv_X_test.npy' : '[adv_4]gabor_40_adv',

    '[adv_8]gabor_20_adv_X_test.npy' : '[adv_8]gabor_20_adv', 
    '[adv_8]gabor_40_adv_X_test.npy' : '[adv_8]gabor_40_adv',

    '[adv_16]gabor_20_adv_X_test.npy' : '[adv_16]gabor_20_adv', 
    '[adv_16]gabor_40_adv_X_test.npy' : '[adv_16]gabor_40_adv',

    '[adv_32]gabor_20_adv_X_test.npy' : '[adv_32]gabor_20_adv', 
    '[adv_32]gabor_40_adv_X_test.npy' : '[adv_32]gabor_40_adv',

    '[doa_5]gabor_20_adv_X_test.npy' : '[doa_5]gabor_20_adv', 
    '[doa_5]gabor_40_adv_X_test.npy' : '[doa_5]gabor_40_adv',

    '[doa_7]gabor_20_adv_X_test.npy' : '[doa_7]gabor_20_adv', 
    '[doa_7]gabor_40_adv_X_test.npy' : '[doa_7]gabor_40_adv',


    ## Elastic
    '[clean]elastic_1.5_adv_X_test.npy' : '[clean]elastic_1.5_adv', 
    '[clean]elastic_2.0_adv_X_test.npy' : '[clean]elastic_2.0_adv',

    '[adv_4]elastic_1.5_adv_X_test.npy' : '[adv_4]elastic_1.5_adv', 
    '[adv_4]elastic_2.0_adv_X_test.npy' : '[adv_4]elastic_2.0_adv',

    '[adv_8]elastic_1.5_adv_X_test.npy' : '[adv_8]elastic_1.5_adv', 
    '[adv_8]elastic_2.0_adv_X_test.npy' : '[adv_8]elastic_2.0_adv',

    '[adv_16]elastic_1.5_adv_X_test.npy' : '[adv_16]elastic_1.5_adv', 
    '[adv_16]elastic_2.0_adv_X_test.npy' : '[adv_16]elastic_2.0_adv',

    '[adv_32]elastic_1.5_adv_X_test.npy' : '[adv_32]elastic_1.5_adv', 
    '[adv_32]elastic_2.0_adv_X_test.npy' : '[adv_32]elastic_2.0_adv',

    '[doa_5]elastic_1.5_adv_X_test.npy' : '[doa_5]elastic_1.5_adv', 
    '[doa_5]elastic_2.0_adv_X_test.npy' : '[doa_5]elastic_2.0_adv',

    '[doa_7]elastic_1.5_adv_X_test.npy' : '[doa_7]elastic_1.5_adv', 
    '[doa_7]elastic_2.0_adv_X_test.npy' : '[doa_7]elastic_2.0_adv',
}




print('Generate border & content edge ......')

for file_path in natural_attack_mapping.keys():

    phase = natural_attack_mapping[file_path]

    border = []
    content = []

    X = np.load('./data/'+file_path)
    num = len(X)

    print('> phase : %s ,  %d samples' % (phase,num))

    for i in range(num):
        img = X[i]
        img = cv2.resize(img,(128,128))
        b_img, c_img = preprocess(img)
        border.append(b_img)
        content.append(c_img)

        if (i+1)%200 == 0:
            print('> %s data :  %d/%d' % (phase,i+1,num))
    
    print('> %s data :  %d/%d' % (phase,i+1,num))


    border = np.array(border)
    content = np.array(content)

    np.save('./data/%s_border.npy' % phase,border)
    np.save('./data/%s_content.npy' % phase,content)

print('Done ..... \n')