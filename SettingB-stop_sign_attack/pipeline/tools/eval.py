import torch
import numpy as np
from pipeline import pipeline
from sensing_signals_data import sensing_data
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-main_sensor', type=str, default='CNN', help='selected main sensor for building KEMLP model')
parser.add_argument('-alpha', type=str, default='0.4', choices=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],\
        help='adversarial ratio, options = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]')
args = parser.parse_args()


main_sensor = args.main_sensor
alpha = args.alpha


n_class = 12

def test(model,data):

    num_each_class = np.zeros(n_class)
    correct_each_class = np.zeros(n_class)
    mis_from_A_to_B = np.zeros((n_class,n_class))

    var, gt =data.sequential_test_batch()
    correct = 0
    tot = 0
    while var is not None:
        this_batch_size = len(gt)

        scores = model.inference(var)
        Y = torch.argmax(scores,dim=1)

        tot += this_batch_size
        for i in range(this_batch_size):
            num_each_class[gt[i]]+=1
            if Y[i] == gt[i]:
                correct+=1
                correct_each_class[gt[i]]+=1
            else:
                mis_from_A_to_B[gt[i]][Y[i]]+=1
        var, gt = data.sequential_test_batch()
    
    print('acc = %d/%d = %f' % (correct,tot,correct/tot))


batch_size = 40




phases = ['clean', 'adv']

print('>>>> Main Sensor : %s\n\n' % main_sensor)

for phase in phases:
    
    print('##### %s data #####' % phase)
    model = torch.load('./ckpt/graph_model_ratio=%s.ckpt' % alpha)
    
    sensing_signals_path = './sensing_signals/%s_signals_with_%s.npy' % (phase, main_sensor)
    gt_path = './sensing_signals/GT_stop_test.npy'

    data = sensing_data(batch_size=batch_size, train_data = None, test_data = [sensing_signals_path, gt_path],)
    test(model,data)
    print('---------------------------------------\n')