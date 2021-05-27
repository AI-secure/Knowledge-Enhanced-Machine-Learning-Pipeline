import torch.nn.functional as F
from model import NEURAL
import torch
from torch import nn
import time
import cv2
import numpy as np
import functools

from fog import FogAttack
from snow import SnowAttack
from gabor import GaborAttack
from jpeg import JPEGAttack
from elastic import ElasticAttack


import os

os.environ['CUDA_VISIBLE_DEVICES']='0,1'

#device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')

n_class = 12
n_iters = 200
resol = 32




print('[Data] Preparing .... ')
X = torch.FloatTensor( np.load('../data/data/raw_feature_test.npy') / 255.0 - 0.5 ).permute(0,3,1,2).cuda()
GT = torch.LongTensor( np.load('../data/data/raw_label_test.npy') ).cuda()
print('[Data] Done .... ')



print('[Model] Preparing .... ')

sub_model = NEURAL(n_class=12,n_channel=3)
ckpt = torch.load('../substitute/sub_pipeline.ckpt', map_location = 'cuda') # black-box sensor attack

sub_model = nn.DataParallel(sub_model)
sub_model = sub_model.cuda()

sub_model.module.load_state_dict(ckpt)

sub_model.eval()
print('[Model] Done .... ')



attack_mapping = {
    'fog_256' : functools.partial(FogAttack, n_iters, 256, (0.001/n_iters)**0.5, resol, scale_each=False),
    'fog_512' : functools.partial(FogAttack, n_iters, 512, (0.001/n_iters)**0.5, resol, scale_each=False),
    'snow_0.25' : functools.partial(SnowAttack, n_iters, 0.25, (0.001/n_iters)**0.5, resol, scale_each=False),
    'snow_0.75' : functools.partial(SnowAttack, n_iters, 0.75, (0.001/n_iters)**0.5, resol, scale_each=False),
    'jpeg_0.125' : functools.partial(JPEGAttack, n_iters, 0.125, 0.125/(n_iters**0.5), resol, scale_each=False, opt='linf'),
    'jpeg_0.25' : functools.partial(JPEGAttack, n_iters, 0.25, 0.25/(n_iters**0.5), resol, scale_each=False, opt='linf'),

    'gabor_20' : functools.partial(GaborAttack, n_iters, 20, (0.001/n_iters)**0.5, resol, scale_each=False),
    'gabor_40' : functools.partial(GaborAttack, n_iters, 40, (0.001/n_iters)**0.5, resol, scale_each=False),

    'elastic_1.5' : functools.partial(ElasticAttack, n_iters, 1.5, 1.5/(n_iters**0.5), resol, scale_each=False),
    'elastic_2.0' : functools.partial(ElasticAttack, n_iters, 2.0, 2.0/(n_iters**0.5), resol, scale_each=False)
}


phase = 'black_box_pipeline'



for attack_type in attack_mapping.keys():

    print('> attack : %s' % attack_type)
    attacker = attack_mapping[attack_type]()

    adv_sample = []
    num = len(GT)
    st = 0
    batch_size = 800

    while st!= num:
        ed = min(st+batch_size,num)

        input_batch = X[st:ed]
        gt_batch = GT[st:ed]
        this_batch_size = len(gt_batch)

        # --------------- attack ---------------
        adv_batch = attacker(sub_model,input_batch,gt_batch,avoid_target=True,scale_eps=False)
        # --------------------------------------

        adv_batch = ( ( adv_batch.cpu().permute(0,2,3,1).detach().numpy() + 0.5 ) * 255 ).astype(np.uint8)
            
        for i in range(this_batch_size):
            adv_sample.append(adv_batch[i])
        st = ed

    adv_sample = np.array(adv_sample, dtype=np.uint8)
    print('[Attack] : Done, adversarial examples are generated.')
    save_path = '../data/data/[%s]%s_adv_X_test.npy' % (phase,attack_type)
    print('[Save] : %s' % save_path)
    np.save(save_path, adv_sample)