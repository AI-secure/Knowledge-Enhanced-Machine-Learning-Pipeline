import os
import math
import cv2
import numpy as np
import torch


class BorderData:

    def __init__(self,pid = 0,batch_size=50,ang_range=10,trans_range=2,shear_range=2):
        super().__init__()

        # pid => the type of border to be detected
        self.class_with_border = {
            0:[0], 
            1:[1],
            2:[2],
            3:[3],
            4:[4,5],
            5:[6],
            6:[7,8,10,11],
            7:[9]
        }

        # Para
        self.batch_size = batch_size
        self.ang_range = ang_range
        self.trans_range = trans_range
        self.shear_range = shear_range

        ############################### Load data ###################################
        for phase in ['train', 'test', 'val']:
            X = np.load('../data/data/border_%s.npy' % phase)
            Y = np.load('../data/data/raw_label_%s.npy' % phase)
            positive = []
            negative = []

            num = len(Y)
            for i in range(num):
                if Y[i] in self.class_with_border[pid]:
                    positive.append(X[i])
                else:
                    negative.append(X[i])

            if phase == 'train':
                self.train_positive = np.array(positive)
                self.train_negative = np.array(negative)
            elif phase == 'test':
                self.test_positive = np.array(positive)
                self.test_negative = np.array(negative)
            else:
                self.val_positive = np.array(positive)
                self.val_negative = np.array(negative)


    def pre_process_image(self,image):
        if image.shape[0]!=32 :
            image = cv2.resize(image,(32,32))
        image = image/255.-.5
        return image

    def transform_image(self,image):
        # Random rotation & translation & shear for data augmentation

        ang_range=self.ang_range
        shear_range=self.shear_range
        trans_range=self.trans_range

        # Rotation
        ang_rot = np.random.uniform(ang_range)-ang_range/2
        rows,cols,ch = image.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

        # Translation
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        # Shear
        pts1 = np.float32([[5,5],[20,5],[5,20]])
        pt1 = 5+shear_range*np.random.uniform()-shear_range/2
        pt2 = 20+shear_range*np.random.uniform()-shear_range/2
        pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
        shear_M = cv2.getAffineTransform(pts1,pts2)

        image = cv2.warpAffine(image,Rot_M,(cols,rows))
        image = cv2.warpAffine(image,Trans_M,(cols,rows))
        image = cv2.warpAffine(image,shear_M,(cols,rows))
        image = self.pre_process_image(image)
        return image

    def gen_extra_data(self,data,n_each):
        # Augment the whole data set, each sample is repeated for n_each time.
        X_arr = []

        num = len(data)
        for i in range(num):
            img = data[i]
            for i_n in range(n_each):
                img_trf = self.transform_image(img)
                X_arr.append(img_trf)

        X_arr = np.array(X_arr,dtype = np.float32())
        np.random.shuffle(X_arr)
        return X_arr

    def data_set_up(self,istrain=True):
        
        ## Positive data ######

        # Test set
        self.test_positive = torch.FloatTensor(
        np.array( [self.pre_process_image(self.test_positive[i]) for i in range(len(self.test_positive))],\
                    dtype = np.float32)
        ).permute(0,3,1,2)
        self.test_positive_batch_cnt = 0
        self.test_positive_num = len(self.test_positive)

        # Val set
        self.val_positive = torch.FloatTensor(
        np.array( [self.pre_process_image(self.val_positive[i]) for i in range(len(self.val_positive))],\
                    dtype = np.float32)
        ).permute(0,3,1,2)
        self.val_positive_batch_cnt = 0
        self.val_positive_num = len(self.val_positive)


        ## Negative data ######

        # Test set
        self.test_negative = torch.FloatTensor(
        np.array( [self.pre_process_image(self.test_negative[i]) for i in range(len(self.test_negative))],\
                    dtype = np.float32)
        ).permute(0,3,1,2)
        self.test_negative_batch_cnt = 0
        self.test_negative_num = len(self.test_negative)

        # Val set
        self.val_negative = torch.FloatTensor(
        np.array( [self.pre_process_image(self.val_negative[i]) for i in range(len(self.val_negative))],\
                    dtype = np.float32)
        ).permute(0,3,1,2)
        self.val_negative_batch_cnt = 0
        self.val_negative_num = len(self.val_negative)


        if istrain:

            num_positive = len(self.train_positive)
            num_negative = len(self.train_negative)
            n_p = num_negative*3 // num_positive

            print('augmenting positive samples ....')
            self.train_positive = torch.FloatTensor(
                self.gen_extra_data(self.train_positive,n_p)
            ).permute(0,3,1,2)
            self.train_positive_num = len(self.train_positive)


            print('augmenting negative samples ....')
            self.train_negative = torch.FloatTensor(
                self.gen_extra_data(self.train_negative,3)
            ).permute(0,3,1,2)
            self.train_negative_num = len(self.train_negative)



    def random_train_batch(self):

        # Create a set of random indices.
        id_positive = np.random.choice(self.train_positive_num,size=self.batch_size,replace=False)
        id_negative = np.random.choice(self.train_negative_num,size=self.batch_size,replace=False)

        # Use the random index to select random images and labels.
        positive_samples = self.train_positive[id_positive]
        negative_samples = self.train_negative[id_negative]

        return positive_samples, negative_samples



    def sequential_test_positive_batch(self):
        if self.test_positive_batch_cnt==-1: # end of the data
            self.test_positive_batch_cnt=0
            return None
        else:
            st = self.test_positive_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,self.test_positive_num)
            if ed == self.test_positive_num:
                self.test_positive_batch_cnt=-1
            else:
                self.test_positive_batch_cnt+=1

            batch = self.test_positive[st:ed]
            return batch

    def sequential_test_negative_batch(self):
        if self.test_negative_batch_cnt==-1: # end of the data
            self.test_negative_batch_cnt=0
            return None
        else:
            st = self.test_negative_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,self.test_negative_num)
            if ed == self.test_negative_num:
                self.test_negative_batch_cnt=-1
            else:
                self.test_negative_batch_cnt+=1

            batch = self.test_negative[st:ed]
            return batch
    
    def sequential_val_positive_batch(self):
        if self.val_positive_batch_cnt==-1: # end of the data
            self.val_positive_batch_cnt=0
            return None
        else:
            st = self.val_positive_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,self.val_positive_num)
            if ed == self.val_positive_num:
                self.val_positive_batch_cnt=-1
            else:
                self.val_positive_batch_cnt+=1

            batch = self.val_positive[st:ed]
            return batch

    def sequential_val_negative_batch(self):
        if self.val_negative_batch_cnt==-1: # end of the data
            self.val_negative_batch_cnt=0
            return None
        else:
            st = self.val_negative_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,self.val_negative_num)
            if ed == self.val_negative_num:
                self.val_negative_batch_cnt=-1
            else:
                self.val_negative_batch_cnt+=1

            batch = self.val_negative[st:ed]
            return batch



    def greeting(self):
        print("****************** Border Data *******************")

if __name__ == "__main__":
    data = BorderData(pid=0)
    data.data_set_up(istrain=True)
    data.random_train_batch()

    cnt = 0
    t = data.sequential_val_positive_batch()
    while t is not None:
        print(cnt)
        cnt+=1
        t = data.sequential_val_positive_batch()