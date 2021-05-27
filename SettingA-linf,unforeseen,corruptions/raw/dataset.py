import os
import math
import cv2
import numpy as np
import torch
import pickle

class DataMain:

    def __init__(self,batch_size = 50, n_class=2, n_each=5, ang_rot=10,trans_rot=2,shear_rot=2):
        
        super().__init__()

        ang_rot,trans_rot,shear_rot
        self.batch_size = batch_size
        self.n_class = n_class
        self.n_each = n_each
        self.ang_rot = ang_rot
        self.trans_rot = trans_rot
        self.shear_rot = shear_rot

        ############################### Load data ###################################
        self.X_train = np.load('../data/data/raw_feature_train.npy')
        self.y_train = np.load('../data/data/raw_label_train.npy')

        self.X_test = np.load('../data/data/raw_feature_test.npy')
        self.y_test = np.load('../data/data/raw_label_test.npy')
        self.X_test = np.array([self.pre_process_image(self.X_test[i]) for i in range(len(self.X_test))],dtype = np.float32)
        self.y_test = np.array(self.y_test,dtype=np.long)

        self.X_val = np.load('../data/data/raw_feature_val.npy')
        self.y_val = np.load('../data/data/raw_label_val.npy')
        self.X_val = np.array([self.pre_process_image(self.X_val[i]) for i in range(len(self.X_val))],dtype = np.float32)
        self.y_val = np.array(self.y_val,dtype=np.long)
        
        self.test_batch_cnt = 0
        self.val_batch_cnt = 0
        self.train_batch_cnt = 0
        ##############################################################################

    def pre_process_image(self,image): 
        image = image/255.-.5
        return image

    def transform_image(self,image,ang_range,shear_range,trans_range):
        # Random rotation & translation & shear for data augmentation

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
    
    def get_index_dict(self,y_train):
        # Returns image indices of each class
        # Assumes that the labels are 0 to N-1
        dict_indices = {}
        ind_all = np.arange(len(y_train))

        for i in range(len(np.unique(y_train))):
            ind_i = ind_all[y_train == i]
            dict_indices[i] = ind_i
        return dict_indices

    def gen_extra_data(self,X_train,y_train,N_classes,n_each,ang_range,shear_range,trans_range,randomize_Var):
        # Augment the whole data set, each sample is repeated for n_each time.
        dict_indices = self.get_index_dict(y_train)
        n_class = len(np.unique(y_train))

        X_arr = []
        Y_arr = []
        n_train = len(X_train)
        
        for i in range(n_train):
            for i_n in range(n_each):
                img_trf = self.transform_image(X_train[i],
                                        ang_range,shear_range,trans_range)
                X_arr.append(img_trf)
                Y_arr.append(y_train[i])

        X_arr = np.array(X_arr,dtype = np.float32())
        Y_arr = np.array(Y_arr,dtype = np.long)

        if (randomize_Var == 1): # shuffle the dataset
            random_state = np.random.get_state()
            np.random.shuffle(X_arr)
            np.random.set_state(random_state)
            np.random.shuffle(Y_arr)
        
        X_arr = torch.FloatTensor(X_arr)
        Y_arr = torch.LongTensor(Y_arr)

        return X_arr,Y_arr
    
    def data_set_up(self,istrain=True):
        if istrain:
            self.image_train_aug, self.y_train_aug = \
                self.gen_extra_data(self.X_train,self.y_train,self.n_class,self.n_each,\
                    self.ang_rot,self.trans_rot,self.shear_rot,1)
            
            self.image_train_aug = self.image_train_aug.permute(0,3,1,2)
        else:
            #self.X_train = np.array([self.pre_process_image(self.X_train[i]) for i in range(len(self.X_train))],dtype = np.float32)
            self.X_train = np.array([self.pre_process_image(self.X_train[i]) for i in range(len(self.X_train))],dtype = np.float32)
            self.y_train = np.array(self.y_train,dtype=np.long)

            self.X_train = torch.FloatTensor(self.X_train)
            self.X_train = self.X_train.permute(0,3,1,2)
            self.y_train = torch.LongTensor(self.y_train)
        
        

        self.X_test = torch.FloatTensor(self.X_test)
        self.X_test = self.X_test.permute(0,3,1,2)
        self.y_test = torch.LongTensor(self.y_test)

        self.X_val = torch.FloatTensor(self.X_val)
        self.X_val = self.X_val.permute(0,3,1,2)
        self.y_val = torch.LongTensor(self.y_val)


        
    def random_train_batch(self):

        # Number of images in the training-set.
        num_images = len(self.image_train_aug)
        # Create a set of random indices.
        idx = np.random.choice(num_images,size=self.batch_size,replace=False)

        # Use the random index to select random images and labels.
        features_batch = self.image_train_aug[idx, :,:,:]
        labels_batch = self.y_train_aug[idx]

        return features_batch, labels_batch
    
    def random_test_batch(self):

        num_images = len(self.X_test)
        idx = np.random.choice(num_images,size=10000,replace=False)

        features_batch = self.X_test[idx, :,:,:]
        labels_batch = self.y_test[idx]

        return features_batch, labels_batch
    
    def sequential_test_batch(self):
        num_images = len(self.X_test)
        if self.test_batch_cnt==-1:
            self.test_batch_cnt = 0
            return None,None
        else:
            st = self.test_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,num_images)
            if ed == num_images:
                self.test_batch_cnt=-1
            else:
                self.test_batch_cnt+=1
            return self.X_test[st:ed,:,:,:], self.y_test[st:ed]

    def sequential_val_batch(self):
        num_images = len(self.X_val)
        if self.val_batch_cnt==-1:
            self.val_batch_cnt = 0
            return None,None
        else:
            st = self.val_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,num_images)
            if ed == num_images:
                self.val_batch_cnt=-1
            else:
                self.val_batch_cnt+=1
            return self.X_val[st:ed,:,:,:], self.y_val[st:ed]
    
    def sequential_train_batch(self):
        num_images = len(self.X_train)
        if self.train_batch_cnt==-1:
            self.train_batch_cnt = 0
            return None,None
        else:
            st = self.train_batch_cnt*self.batch_size
            ed = min(st+self.batch_size,num_images)
            if ed == num_images:
                self.train_batch_cnt=-1
            else:
                self.train_batch_cnt+=1
            return self.X_train[st:ed,:,:,:], self.y_train[st:ed]
        
    
    def greeting(self):
        print("****************** raw data *******************")


if __name__ == "__main__":
    data = DataMain()
    data.data_set_up(istrain=False)
    print('training : %d' % len(data.X_train))
    print('testing : %d' % len(data.X_test))
    print('validation : %d' % len(data.X_val))