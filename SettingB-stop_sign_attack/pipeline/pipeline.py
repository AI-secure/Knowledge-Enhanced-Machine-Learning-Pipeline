import torch
import numpy as np 

"""
    Probabilistic reasoning component that makes final decision based on sensing signals.
    "deterministic sensor" corresponds to the "consequential sensor" in our paper.
    "conservative sensor" corresponds to the "antecedential sensor" in our paper.
"""
class pipeline:
    
    def __init__(self,n_class):
        self.n_class = n_class
        self.main_sensor = None
        self.conservative_sensor = [ [] for i in range(n_class) ]
        self.deterministc_sensor = [ [] for i in range(n_class) ]
        self.num_conservative = np.zeros(n_class,dtype = np.int)
        self.num_deterministic = np.zeros(n_class, dtype=np.int)
        self.id_conservative = [ [] for i in range(n_class) ]
        self.id_deterministic = [ [] for i in range(n_class) ]
        self.id_main = 0
        self.already_built = False
    
    def set_main_sensor(self,main_sensor):
        self.main_sensor = main_sensor
    
    def add_conservative_sensor(self,class_id,var_id,sensor):
        self.conservative_sensor[class_id].append(sensor)
        self.id_conservative[class_id].append(var_id)
        self.num_conservative[class_id]+=1
    
    def add_deterministic_sensor(self,class_id,var_id,sensor):
        self.deterministc_sensor[class_id].append(sensor)
        self.id_deterministic[class_id].append(var_id)
        self.num_deterministic[class_id]+=1
    
    def build_graph(self):
        if self.main_sensor is None:
            return False
        
        cnt = 1
        
        # id for conservative sensors
        for i in range(self.n_class):
            cnt += self.num_conservative[i]
            
        
        # id for deterministic sensors
        for i in range(self.n_class):
            cnt += self.num_deterministic[i]
            
        
        self.num_sensor = cnt

        # weight for each sensor
        self.weight = torch.ones( int(self.num_sensor) )
        self.weight.requires_grad = True

        self.already_built = True

        #print('conservative_id : ',self.id_conservative)
        #print('deterministic_id : ',self.id_deterministic)
        return True
    
    def inference(self,X):
        # X : sensor variable  ==> batch_size x num_of_sensors
        #print(X)
        score = []
        for i in range(self.n_class):
            #print('## class-%d' % i)

            factor_main = (X[:, 0] == i) * self.weight[0]
            factor_conservative = torch.zeros(factor_main.shape)
            factor_deterministic = torch.zeros(factor_main.shape)

            #print('conservative --->')
            for j in range(self.num_conservative[i]):
                sid = self.id_conservative[i][j]
                factor_conservative+= (X[:,sid]==1) * self.weight[sid]
                #print('sid : %d, result : ' % sid, X[:,sid]==1)
            
            #print('deterministic --->')
            for j in range(self.num_deterministic[i]):
                sid = self.id_deterministic[i][j]
                factor_deterministic+= -1 * ( (X[:,sid]==0) * self.weight[sid] )
                #print('sid : %d, result : ' % sid, X[:,sid]==0)
            
            
            #print('factor conservative :',factor_conservative)
            #print('factor deterministic :',factor_deterministic)
            #print('factor main :',factor_main)
            

            result = torch.exp(factor_main + factor_conservative + factor_deterministic)
            result = result.unsqueeze(1)

            score.append( result )
        #exit(0)
        score = torch.cat(score,dim=1)
        return score

    def step(self,step_size): # maximize the likelihood
        
        self.weight = self.weight + step_size*self.weight.grad
        for i in range(self.num_sensor):
            if self.weight[i]<0:
                self.weight[i] = 0
        #self.weight[0] = 1 # fix the weight for the main sensor
        
        self.weight = torch.tensor(self.weight.data, requires_grad=True)

        if self.weight.grad is not None:
            self.weight.grad.data.fill_(0)