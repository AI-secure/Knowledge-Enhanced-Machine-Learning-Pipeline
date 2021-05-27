import os
import time

import numpy as np
import torch
import cv2
from dataset import DigitData
from model import Model


batch_size = 200

number = { # class id -> content number sequence
    0 : [0,10,10,10,10,10],
    1 : [0,10,10,10,10,10],
    2 : [0,10,10,10,10,10],
    3 : [0,10,10,10,10,10],
    4 : [0,10,10,10,10,10],
    5 : [0,10,10,10,10,10],
    6 : [0,10,10,10,10,10],
    7 : [0,10,10,10,10,10],
    8 : [2,5,0,10,10,10],
    9 : [0,10,10,10,10,10],
    10 : [2,2,0,10,10,10],
    11 : [3,1,2,0,10,10]
}

def invert_number(labels):
    if labels[0]==0:
        return -1
    elif labels[1]==5:
        return 8
    elif labels[1]==2:
        return 10
    elif labels[1]==1:
        return 11

def pre_process_image(image):
    image = cv2.resize(image,(54,54))
    image = image/255.-.5
    return image


print('preparing model ...')
model = Model()
model.restore('digit_log/model-16000.pth')
model.cuda()
model.eval()
print('done.')

print('preparing data ...')
X = np.load('../data/data/patch_content_adv.npy')
y = np.load('../data/data/raw_label_test.npy')
labels = []
num = len(y)
for i in range(num):
    c = y[i]
    labels.append(number[c])
y = torch.LongTensor(labels).cuda()
X = torch.FloatTensor([pre_process_image(X[i]) for i in range(len(X))]).unsqueeze(1).cuda()
print('done.')



TP = np.zeros(12)
FP = np.zeros(12)
n_each_class = np.zeros(12)

st = 0
while st!= num:
    ed = min(st+batch_size,num)
    input_batch = X[st:ed]
    GT = y[st:ed]

    length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(input_batch)
    length_prediction = length_logits.max(1)[1]
    digit1_prediction = digit1_logits.max(1)[1]
    digit2_prediction = digit2_logits.max(1)[1]
    digit3_prediction = digit3_logits.max(1)[1]
    digit4_prediction = digit4_logits.max(1)[1]
    digit5_prediction = digit5_logits.max(1)[1]



    this_batch_size = len(GT)

    for i in range(this_batch_size):

        inv_id = invert_number(GT[i])
        if inv_id!=-1 : n_each_class[inv_id]+=1

        for k in [8,10,11]:
            gt_k = number[k]
            if gt_k[0] != length_prediction[i]: continue

            mark = True
            for item in range(1,6):
                if gt_k[item] != eval('digit%d_prediction' % item)[i] :
                    mark = False
            if mark :
                if inv_id == k: TP[k]+=1
                else: FP[k]+=1

    st = ed

    print('proecss : %d/%d' % (st,num))


for i in [8,10,11]:
    print('### Class - %d' % i )
    print(' Recall = %d/%d = %f' % (TP[i],n_each_class[i],TP[i]/n_each_class[i]))
    print(' Precision = %d/%d = %f' % (TP[i], TP[i]+FP[i], TP[i]/(TP[i]+FP[i])))
    print('--------------------------------------------------------')