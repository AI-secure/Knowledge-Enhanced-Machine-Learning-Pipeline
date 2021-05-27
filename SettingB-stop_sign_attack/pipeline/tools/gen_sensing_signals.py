import numpy as np
import torch
import os
import json
from model import NEURAL,Model
from var_generator import var_generator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-enable_cuda', type=bool, default=True, help='whether use cuda device')
parser.add_argument('-gpu_id', type=int, default=0, help='id number of the gpu device')
parser.add_argument('-main_sensor', type=str, default='CNN', help='selected main sensor for building KEMLP model')
args = parser.parse_args()


enable_cuda = args.enable_cuda
gpu_id = args.gpu_id
main_sensor = args.main_sensor

signal_dir = 'sensing_signals/'
if not os.path.exists(signal_dir):
    os.mkdir(signal_dir)

if enable_cuda:
    device = torch.device('cuda:%d' % gpu_id)
else:
    device = torch.device('cpu')

main_sensor_list = {
    # standard CNN main sensor
    'CNN' : './sensor/main/main.ckpt',
    # adversarially trained CNN main sensor
    'CNN_adv_4' : './sensor/main/main_adv_4.ckpt',
    'CNN_adv_8' : './sensor/main/main_adv_8.ckpt',
    'CNN_adv_16' : './sensor/main/main_adv_16.ckpt',
    'CNN_adv_32' : './sensor/main/main_adv_32.ckpt',
    # DOA trained CNN main sensor
    'CNN_doa_5' : './sensor/main/main_doa_5.ckpt',
    'CNN_doa_7' : './sensor/main/main_doa_7.ckpt'
}

if main_sensor not in main_sensor_list.keys():
    print('Invalid main sensor. Supported main sensor types:')
    print(main_sensor_list.keys())
    exit(0)
else:
    main_sensor_name = main_sensor
    main_sensor = main_sensor_list[main_sensor]

n_class = 12
batch_size = 100


##### Sensors & Variables ##################
digit_sensor = './sensor/content/digit.pth'

con_sensors_for_class = { 
    './sensor/border/type_0_con.ckpt' : [0], 
    './sensor/border/type_1_con.ckpt' : [1], 
    './sensor/border/type_2_con.ckpt' : [2], 
    './sensor/border/type_3_con.ckpt' : [3], 
    './sensor/content/class_4.ckpt' : [4],
    './sensor/content/class_5.ckpt' : [5],
    './sensor/border/type_5_con.ckpt' : [6], 
    './sensor/content/class_7.ckpt' : [7],
    './sensor/content/digit.pth' : [8,10,11], # digit recognition model for speed-limit 50/20/120
    './sensor/border/type_7_con.ckpt' : [9]
}

det_sensors_for_class = {
    './sensor/border/type_0_det.ckpt' : [0], 
    './sensor/border/type_1_det.ckpt' : [1],
    './sensor/border/type_2_det.ckpt' : [2],
    './sensor/border/type_3_det.ckpt' : [3],
    './sensor/border/type_4_det.ckpt' : [4,5],
    './sensor/border/type_5_det.ckpt' : [6],
    './sensor/border/type_6_det.ckpt' : [7,8,10,11],
    './sensor/border/type_7_det.ckpt' : [9]
}

aux_sensor_map = { # sensor_ckpt : [ (class_id, sensor_variable_id), ... ]
    './sensor/border/type_0_con.ckpt' : [], 
    './sensor/border/type_1_con.ckpt' : [], 
    './sensor/border/type_2_con.ckpt' : [], 
    './sensor/border/type_3_con.ckpt' : [], 
    './sensor/content/class_4.ckpt' : [],
    './sensor/content/class_5.ckpt' : [],
    './sensor/border/type_5_con.ckpt' : [], 
    './sensor/content/class_7.ckpt' : [],
    './sensor/content/digit.pth' : [],
    './sensor/border/type_7_con.ckpt' : [],
    './sensor/border/type_0_det.ckpt' : [], 
    './sensor/border/type_1_det.ckpt' : [],
    './sensor/border/type_2_det.ckpt' : [],
    './sensor/border/type_3_det.ckpt' : [],
    './sensor/border/type_4_det.ckpt' : [],
    './sensor/border/type_5_det.ckpt' : [],
    './sensor/border/type_6_det.ckpt' : [],
    './sensor/border/type_7_det.ckpt' : []
}


cnt = 1

for i in range(n_class):
    for item in con_sensors_for_class.keys():
        if i in con_sensors_for_class[item]: # if sensor 'item' is for class-i, then create a variable for it.
            aux_sensor_map[item].append( (i,cnt) )
            cnt+=1

for i in range(n_class):
    for item in det_sensors_for_class.keys():
        if i in det_sensors_for_class[item]:
            aux_sensor_map[item].append( (i,cnt) )
            cnt+=1

print('num_variable = %d' % cnt)

with open(signal_dir+'sensor_map.json','w') as file_obj:
    json.dump(aux_sensor_map,file_obj)
#######################################################




############# data ##########################################

phase_to_data = { # [GT, raw_img, border_img, content_img]
    # clean stop sign samples
    'clean' : ['stop_sign_label.npy', 'stop_sign_X_test.npy', 'stop_sign_border_test.npy', 'stop_sign_content_test.npy'],
    # adversarial stop sign samples
    'adv' : ['stop_sign_label.npy', 'stop_sign_adv_X_test.npy', 'stop_sign_adv_border_test.npy', 'stop_sign_adv_content_test.npy']
}

phase_set = phase_to_data.keys()
data_dir = '../data/data/'
#############################################################





my_generator = var_generator(main_sensor_path=main_sensor, digit_sensor_path=digit_sensor, n_class = n_class,\
    batch_size = batch_size, var_cnt=cnt, con_list = con_sensors_for_class.keys(), det_list = det_sensors_for_class.keys(),\
        aux_list = aux_sensor_map, device = device)

for phase in phase_to_data.keys():

    print('------------------- phase : %s --------------------------' % phase)
    signal_list = phase_to_data[phase]
    for k,item in enumerate(signal_list):
        signal_list[k] = data_dir+item
    VARS = my_generator.gen_vars(signal_list)
    np.save(signal_dir + '%s_signals_with_%s.npy' % (phase, main_sensor_name),VARS)

    print('\n\n\n\n\n\n')