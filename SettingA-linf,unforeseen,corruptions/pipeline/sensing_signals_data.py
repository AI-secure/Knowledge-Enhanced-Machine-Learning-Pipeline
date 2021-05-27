import numpy as np 
import torch

class sensing_data:

    def __init__(self, train_data, test_data, batch_size=50):

        super().__init__()

        self.batch_size = batch_size
        
        if train_data is not None:
            self.vars_train = torch.FloatTensor( np.load(train_data[0]) )
            self.gt_train = np.load(train_data[1])
            self.num_train = len(self.gt_train)

        if test_data is not None:
            self.vars_test = torch.FloatTensor( np.load(test_data[0]) )
            self.gt_test = np.load(test_data[1])
            self.num_test = len(self.gt_test)
            self.test_batch_cnt = 0
    
    def random_train_batch(self):
        # Create a set of random indices.
        idx = np.random.choice(self.num_train,size=self.batch_size,replace=False)
        return self.vars_train[idx], self.gt_train[idx]

    def adv_random_train_batch(self):
        idx = np.random.choice(self.num_train,size=self.batch_size,replace=False)
        adv_var = torch.clone(self.vars_train[idx])
        adv_var[:,0] = torch.randint(10,[self.batch_size])
        return adv_var, self.gt_train[idx]
    
    def sequential_test_batch(self):
        if self.test_batch_cnt==-1:
            self.test_batch_cnt=0
            return None,None
        
        st = self.test_batch_cnt*self.batch_size
        ed = min(st+self.batch_size,self.num_test)

        vars_batch = self.vars_test[st:ed]
        gt_batch = self.gt_test[st:ed]

        if ed == self.num_test:
            self.test_batch_cnt=-1
        else:
            self.test_batch_cnt+=1
        
        return vars_batch, gt_batch

