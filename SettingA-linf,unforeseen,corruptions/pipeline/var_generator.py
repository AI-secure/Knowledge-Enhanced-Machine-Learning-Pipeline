import numpy as np
import torch
import os
import json
from model import NEURAL,Model
import cv2


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

class var_generator:
    
    def __init__(self,main_sensor_path,digit_sensor_path,n_class,\
        batch_size,var_cnt,con_list,det_list,aux_list,device):
        super().__init__()

        self.device = device
        self.var_cnt = var_cnt
        self.n_class = n_class
        self.batch_size = batch_size
        self.con_list = con_list # list of conservative sensors
        self.det_list = det_list # list of deterministic sensors
        self.aux_list = aux_list # list of auxiliary sensors

        #print('create end-to-end main sensor...')
        model = NEURAL(n_class=n_class,n_channel=3) 
        ckpt = torch.load(main_sensor_path,map_location = 'cpu')
        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        self.main_sensor = model
        #print('done ..')

        #print('create digit sensor...')
        model = Model()
        model.restore(digit_sensor_path)
        model = model.to(device)
        model.eval()
        self.digit_sensor = model
        #print('done.')

        #print('create border sensor...')
        model = NEURAL(n_class=2,n_channel=3)
        model = model.to(device)
        model.eval()
        self.border_sensor = model
        #print('done.')

        #print('create content sensor...')
        model = NEURAL(n_class=2,n_channel=1)
        model = model.to(device)
        model.eval()
        self.content_sensor = model
        #print('done.')
        


    def gen_vars(self,signal_list):
        
        device = self.device
        batch_size = self.batch_size
        con_list = self.con_list
        det_list = self.det_list
        aux_list = self.aux_list
        n_class = self.n_class

        # signal_list : [GT, raw_img, border_img, content_img]
        # --- load data ---
        GT = np.load(signal_list[0])
        num = len(GT)

        raw_img = torch.FloatTensor( np.load(signal_list[1]) ).permute(0,3,1,2)/255.-.5
        #border_img = torch.FloatTensor( np.load(signal_list[2]) ).permute(0,3,1,2)
        #content_img = torch.FloatTensor( np.load(signal_list[3]) ).unsqueeze(dim=1)
        
        border_img = np.load(signal_list[2])
        border_img = torch.FloatTensor( [ cv2.resize(border_img[i],(32,32)) for i in range(num) ] ).permute(0,3,1,2)/255.-.5

        content_img = np.load(signal_list[3])
        content_img_32 = torch.FloatTensor( [ cv2.resize(content_img[i],(32,32)) for i in range(num) ] ).unsqueeze(dim=1)/255.-.5
        content_img_54 = torch.FloatTensor( [ cv2.resize(content_img[i],(54,54)) for i in range(num) ] ).unsqueeze(dim=1)/255.-.5
        
        
        
        Vars = np.zeros( (num,self.var_cnt) , dtype=np.int) # num x var_cnt
        

        ## main sensor 
        correct = 0
        #print('--- query : main sensor ---')
        st = 0
        ed = min(st+batch_size,num)
        while(st!=num):
            batch = raw_img[st:ed].to(device)
            Y = self.main_sensor(batch)
            Y = torch.argmax(Y,dim=1)
            for i in range(st,ed):
                Vars[i][0] = int(Y[i-st])
                if Y[i-st] == GT[i]: correct+=1
            st = ed
            ed = min(st+batch_size,num)
        print('[main_sensor] acc = %d/%d = %f' % (correct,num,correct/num))
        #print('######################\n\n')


        ## Conservative sensors
        for item in con_list:


            #print('--- query :', item,'---')
            model = None
            is_digit = False
            related_var_id = aux_list[item]

            if item.endswith('digit.pth'):
                
                TP = np.zeros(n_class)
                FP = np.zeros(n_class)
                n_each_class = np.zeros(n_class)

                model = self.digit_sensor
                model.eval()
                X = content_img_54
                is_digit = True
            else:
                
                TP = 0
                FP = 0
                NP = 0
                
                if item.count('content') > 0: 
                    model = self.content_sensor
                    X = content_img_32
                else: 
                    model = self.border_sensor
                    X = border_img
                ckpt = torch.load(item,map_location=device)
                model.load_state_dict(ckpt)
                model.eval()
            
            st = 0
            ed = min(st+batch_size,num)

            while(st!=num):
                batch = X[st:ed].to(device)

                if is_digit: # Use digit detector

                    length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(batch)
                    length_prediction = length_logits.max(1)[1]
                    digit1_prediction = digit1_logits.max(1)[1]
                    digit2_prediction = digit2_logits.max(1)[1]
                    digit3_prediction = digit3_logits.max(1)[1]
                    digit4_prediction = digit4_logits.max(1)[1]
                    digit5_prediction = digit5_logits.max(1)[1]

                    for i in range(st,ed):

                        n_each_class[GT[i]]+=1

                        img = X[i]+0.5
                        if img.sum() < 10:  # conservative => if segmentation fails, return 'no digit detected'
                            length_prediction[i-st] = 0
                        
                        number_detected = False
                        detected_class = -1

                        for k in [8,10,11]: # which number is detected ? 
                            gt_k = number[k]
                            if gt_k[0] != length_prediction[i-st]: continue

                            match = True
                            for digit_item in range(1,6):
                                if gt_k[digit_item] != eval('digit%d_prediction' % digit_item)[i-st] :
                                    match = False
                                    break
                            
                            if match :
                                number_detected = True
                                detected_class = k
                                break
                        
                        for temp in related_var_id:

                            if not number_detected: # no number detected , all digit variables set to 0
                                Vars[i][temp[1]] = 0
                            else:
                                if temp[0] == detected_class: 
                                    Vars[i][temp[1]] = 1
                                    if detected_class == GT[i]: TP[detected_class]+=1
                                    else: FP[detected_class]+=1
                                else : 
                                    Vars[i][temp[1]] = 0
                            
                else: # use binary content classifier

                    Y = model(batch)
                    Y = torch.argmax(Y,dim=1)

                    for i in range(st,ed):
                        img = X[i]+0.5
                        if img.sum() < 10: # conservative : segmentation fails => reject
                            Y[i-st]=0
                        
                        MARK = False
                        for temp in related_var_id:
                            Vars[i][temp[1]] = int(Y[i-st])
                            if not MARK and temp[0] == GT[i]: # is_positive = true
                                NP+=1 # num positive ++
                                MARK = True
                                if Y[i-st] == 1: # true positive ++
                                    TP+=1
                        if not MARK and Y[i-st] == 1: # is_positive = false ^ prediction = positive
                            FP+=1 # false positive ++
                st = ed
                ed = min(st+batch_size,num)
            

            if is_digit :

                if n_each_class[8]!=0: recall = TP[8]/n_each_class[8]
                else : recall = 0

                if (TP[8]+FP[8])!=0 : precision = TP[8]/(TP[8]+FP[8])
                else: precision = 0

                #print('[speed limit 50km/h] recall = %d/%d = %f, precision = %d/%d = %f' \
                #    % (TP[8],n_each_class[8], recall,TP[8],TP[8]+FP[8],precision) )

                if n_each_class[10]!=0: recall = TP[10]/n_each_class[10]
                else : recall = 0

                if (TP[10]+FP[10])!=0 : precision = TP[10]/(TP[10]+FP[10])
                else: precision = 0

                #print('[speed limit 20km/h] recall = %d/%d = %f, precision = %d/%d = %f' \
                #    % (TP[10],n_each_class[10],recall,TP[10],TP[10]+FP[10],precision) )
                
                if n_each_class[11]!=0: recall = TP[11]/n_each_class[11]
                else : recall = 0

                if (TP[11]+FP[11])!=0 : precision = TP[11]/(TP[11]+FP[11])
                else: precision = 0

                #print('[speed limit 120km/h] recall = %d/%d = %f, precision = %d/%d = %f' \
                #    % (TP[11],n_each_class[11],recall,TP[11],TP[11]+FP[11],precision) )
                
            else:
                if NP!=0: recall = TP/NP
                else : recall = 0

                if TP+FP!=0 : precision = TP/(TP+FP)
                else: precision = 0

                #print('recall = %d/%d = %f, precision = %d/%d = %f' % (TP,NP,recall,TP,TP+FP,precision) )
            
            #print('######################\n\n')
        

        ## Deterministic sensors

        for item in det_list:

            #print('--- query :', item,'---')
            model = self.border_sensor
            ckpt = torch.load(item,map_location=device)
            model.load_state_dict(ckpt)
            model.eval()
            related_var_id = aux_list[item]

            TP = 0
            TN = 0
            NP = 0
            NN = 0

            X = border_img

            st = 0
            ed = min(st+batch_size,num)
            while(st!=num):
                batch = X[st:ed].to(device)
                Y = model(batch)
                Y = torch.argmax(Y,dim=1)

                for i in range(st,ed):
                    img = X[i]+0.5
                    if img.sum() < 10: # deterministic : segmentation fails => accept
                        Y[i-st]=1
                        
                    MARK = False
                    for temp in related_var_id:
                        Vars[i][temp[1]] = int(Y[i-st])
                        if not MARK and temp[0] == GT[i]: # is_positive = true
                            NP+=1 # num positive ++
                            MARK = True
                            if Y[i-st] == 1: # true positive ++
                                TP+=1
                    if not MARK: #is_positive = false
                        NN+=1 # num negative ++
                        if Y[i-st] == 0: # prediction = negative
                            TN+=1 # true negative ++
                st = ed
                ed = min(st+batch_size,num)
            
            if NP!=0: recall = TP/NP
            else: recall = 0

            if NN!=0 : tnr = TN/NN
            else: tnr = 0
            
            #print('recall = %d/%d = %f, true_negative_rate = %d/%d = %f' % (TP,NP,recall,TN,NN,tnr ) )
            #print('######################\n\n')
        
        return Vars