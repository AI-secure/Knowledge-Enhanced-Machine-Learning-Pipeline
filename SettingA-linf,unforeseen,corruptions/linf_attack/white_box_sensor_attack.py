import torch.nn.functional as F
from model import NEURAL
import torch
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-enable_cuda', type=bool, default=True, help='whether use cuda device')
parser.add_argument('-gpu_id', type=int, default=0, help='id number of the gpu device')
args = parser.parse_args()


enable_cuda = args.enable_cuda
gpu_id = args.gpu_id


if enable_cuda:
    device = torch.device('cuda:%d' % gpu_id)
else:
    device = torch.device('cpu')



def pgd_attack_random(model, images, labels, eps=1, alpha=1, iters=40, randomize=True):
    """ Construct L_inf adversarial examples on the examples X """
    model.eval()
    labels = labels.to(device)
    if randomize:
        delta = torch.rand_like(images, requires_grad=True).to(device)
        delta.data = delta.data * 2 * eps - eps
        delta.data = (delta.data + images ).clamp(-0.5,0.5)-(images)
    else:
        delta = torch.zeros_like(images, requires_grad=True).to(device)
    
    for t in range(iters):
        loss = torch.nn.CrossEntropyLoss()(model(images + delta ), labels)
        loss.backward()
        
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps,eps)
        delta.data = (delta.data + images ).clamp(-0.5,0.5)-(images)
        delta.grad.zero_()
    
    return delta+images

n_class = 12



print('[Data] Preparing .... ')
X = torch.FloatTensor( np.load('../data/data/raw_feature_test.npy') / 255.0 - 0.5 ).permute(0,3,1,2).to(device)
GT = torch.LongTensor( np.load('../data/data/raw_label_test.npy') ).to(device)
print('[Data] Done .... ')






print('[Model] Preparing .... ')
main_sensor = NEURAL(n_class=12,n_channel=3)
main_sensor = main_sensor.to(device)
main_sensor.eval()
print('[Model] Done .... ')



for adv_trained_model in ['clean', 'adv_4', 'adv_8', 'adv_16', 'adv_32', 'doa_5', 'doa_7']: # 7 baselines

    print('>>>>>>>>>> model : %s' % adv_trained_model)
    
    if adv_trained_model == 'clean':
        ckpt = torch.load('../pipeline/sensor/main/main.ckpt',map_location=device)
    else:
        ckpt = torch.load('../pipeline/sensor/main/main_%s.ckpt' % adv_trained_model ,map_location=device)
    main_sensor.load_state_dict(ckpt)


    print('[Attack] : Start to attack.')

    adv_type = ['pgd_4', 'pgd_8', 'pgd_16','pgd_32']

    name_to_value = {
        'pgd_4' : 4/255, 
        'pgd_8' : 8/255, 
        'pgd_16' : 16/255,
        'pgd_32' : 32/255
    }

    for adv in adv_type:

        print('---------- Adv : %s --------------' % adv)

        success = 0
        adv_sample = []
        batch_size = 50
        num = len(GT)
        st = 0

        while st!= num:
            ed = min(st+batch_size,num)
            input_batch = X[st:ed]
            gt_batch = GT[st:ed]
            adv_batch = pgd_attack_random(main_sensor,input_batch,gt_batch,eps=name_to_value[adv],alpha=1/255,iters=40,randomize=False)
            

            Y_adv = main_sensor(adv_batch)
            Y_adv = torch.argmax(Y_adv,dim=1)

            adv_batch = ( ( adv_batch.cpu().permute(0,2,3,1).detach().numpy() + 0.5 ) * 255 ).astype(np.uint8)
            this_batch_size = len(gt_batch)
            for i in range(this_batch_size):
                adv_sample.append(adv_batch[i])
                if Y_adv[i]!=gt_batch[i]: success+=1
            st = ed

        print('process : %d/%d, accuracy=%d/%d, success=%d/%d' % (st,num,num-success,num,success,num))

        adv_sample = np.array(adv_sample, dtype=np.uint8)
        print('[Attack] : Done, adversarial examples are generated.')
        
        np.save('../data/data/[%s]%s_adv_X_test.npy' % (adv_trained_model,adv), adv_sample)