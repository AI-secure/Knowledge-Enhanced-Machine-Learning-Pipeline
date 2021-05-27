import torch
import numpy as np
from pipeline import pipeline
from pipeline_dataset import pipeline_data
import json


def test(model,data):
    var, gt =data.sequential_test_batch()
    correct = 0
    tot = 0
    while var is not None:
        this_batch_size = len(gt)

        scores = model.inference(var)
        Y = torch.argmax(scores,dim=1)

        
        tot += this_batch_size
        for i in range(this_batch_size):
            if Y[i] == gt[i]:
                correct+=1
        var, gt = data.sequential_test_batch()
    
    print('acc = %d/%d = %f' % (correct,tot,correct/tot))


batch_size = 50
n_iter = 4000
n_class = 12

with open('./statistics/sensor_map.json', 'r') as f:
    sensor_map = json.load(f)



for K in range(11):
    ratio = K/10.0
    print('## ratio = %f' % ratio)

    model = pipeline(n_class=n_class)
    model.set_main_sensor('./sensor/main/main.ckpt')

    clean_ratio = 2*(1-ratio)
    adv_ratio = 2*ratio

    for item in sensor_map.keys():
        var_list = sensor_map[item]
        isconservative = not item.endswith('det.ckpt')
        
        if isconservative:
            for sid in var_list:
                model.add_conservative_sensor(class_id=sid[0],var_id=sid[1],sensor=item)
        else:
            for sid in var_list:
                model.add_deterministic_sensor(class_id=sid[0],var_id=sid[1],sensor=item)


    if not model.build_graph():
        print('Failed to build graph. Terminaite ..')
        exit(0)

    
    # use the sensing signals of validation set to train the weights --- because the sensors performs nearly perfect on the training set due to overftting
    data = pipeline_data(batch_size=batch_size, train_data = './statistics/vars_val.npy', \
        train_GT='./statistics/GT_val.npy', test_data = './statistics/vars_test.npy', test_GT='./statistics/GT_test.npy')

    for i in range(n_iter):

        ## Clean data
        var, gt = data.random_train_batch()

        scores = model.inference(var) # batch x num_class

        sum_scores = scores.sum(dim=1)
        for j in range(batch_size):
            scores[j,:] /= sum_scores[j]
        likelihood = scores[0][gt[0]]
        for j in range(1,batch_size):
            likelihood+=scores[j][gt[j]]
        likelihood/=batch_size

        ## Adversarial data
        var, gt = data.adv_random_train_batch()
        scores = model.inference(var) # batch x num_class
        sum_scores = scores.sum(dim=1)
        for j in range(batch_size):
            scores[j,:] /= sum_scores[j]
        likelihood_adv = scores[0][gt[0]]
        for j in range(1,batch_size):
            likelihood_adv+=scores[j][gt[j]]
        likelihood_adv/=batch_size

        likelihood = clean_ratio*likelihood + adv_ratio*likelihood_adv

        likelihood.backward()
        model.step(step_size=0.1)

        
    print('weight : ',model.weight)
    test(model,data)
    torch.save(model,'./ckpt/graph_model_ratio=%.1f.ckpt' % ratio)