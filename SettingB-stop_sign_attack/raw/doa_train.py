import torch 
from model import NEURAL
from dataset import DataMain
import time
import argparse
import os
import ROA

parser = argparse.ArgumentParser()
parser.add_argument('-gpu_id', type=int, required=True, help='id number of the gpu device')
parser.add_argument('-patch_size', type=int, required=True, help='noise magnitude in adversarial training')
parser.add_argument('-train_id', type=int, required=True, help='id of training task')
args = parser.parse_args()


gpu_id = args.gpu_id
patch_size = args.patch_size
train_id = args.train_id

print('gpu-id : ', gpu_id)
print('patch_size : ', patch_size)
print('train-id : ', train_id)


device = torch.device('cuda:%d' % gpu_id if torch.cuda.is_available() else 'cpu')

lr_rate = 0.01
batch_size = 200
n_iters = 5000
main_sensor_path = '../pipeline/sensor/main/main.ckpt'

ckpt_root = 'adv_train_doa_%dx%d_ckpt_%d/' % (patch_size,patch_size,train_id)
if not os.path.exists(ckpt_root):
    os.mkdir(ckpt_root)

print('saving path :',ckpt_root)


def test(data,model,attacker):
    
    correct_adv = 0
    correct = 0
    tot = 0
    X,GT = data.sequential_val_batch()
    while X is not None:

        X = X.to(device)
        GT = GT.to(device)
        
        X_adv = attacker.exhaustive_search(X,GT,0.05,30,patch_size,patch_size,2,2,False)

        Y = model(X)
        Y = torch.argmax(Y,dim=1)

        Y_adv = model(X_adv)
        Y_adv = torch.argmax(Y_adv,dim=1)

        this_batch_size = len(Y)
        
        for i in range(this_batch_size):
            tot+=1
            if GT[i] == Y[i]:
                correct+=1
            if GT[i] == Y_adv[i]:
                correct_adv+=1
        X,GT = data.sequential_val_batch()
    
    print('acc = %d/%d, adv_acc = %d/%d' % (correct,tot, correct_adv,tot))

    return correct_adv/tot





print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
print('start from : ', main_sensor_path)
model = NEURAL(n_class=43,n_channel=3) 
ckpt = torch.load(main_sensor_path,map_location = 'cpu')
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
attacker = ROA.ROA(base_classifier=model,size=32,device=device)
adv_acc = test(data,model,attacker)
print('[Load successfully] initial adv accuracy = %f' % adv_acc)
print('[Model] Done .... ')




loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay = 1e-4)

st = time.time()

model.train()
max_acc = 0
save_n = 0
stable_iter = 0


print('[Training] Starting ...')
for i in range(n_iters):

    X,GT = data.random_train_batch()
    
    X = X.to(device)
    GT = GT.to(device)

    X_adv = attacker.exhaustive_search(X,GT,0.05,30,patch_size,patch_size,2,2,False)

    X_adv = torch.cat([X_adv,X],dim=0)
    GT = torch.cat([GT,GT],dim=0)

    model.train()

    Y = model(X_adv)
    loss = loss_f(Y,GT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    now = (time.time() - st) / 60.0

    if (i+1) % 10 ==0:
        print('[process: %d/%d] loss = %f' % (i+1,n_iters,loss))

    if (i+1) % 1000 == 0 :
        print(' ### Eval ###')
        print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))
        model.eval()
        
        score = test(data,model,attacker)

        if score>max_acc:
            print('[save]..')
            max_acc = score
            stable_iter = 0
            torch.save(model.state_dict(), ckpt_root+'model_' + str(save_n) + '_adv_acc=%f.ckpt'%(score))
            save_n+=1
        else:
            stable_iter += 1
            if stable_iter == 10:
                print('Stable ... Training END ..')
                break
        model.train()


print('[Training] Done ...')