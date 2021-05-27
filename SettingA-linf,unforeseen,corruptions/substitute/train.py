from dataset import DataSubstitute
from model import NEURAL
import torch
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-enable_cuda', type=bool, default=True, help='whether use cuda device')
parser.add_argument('-gpu_id', type=int, default=0, help='id number of the gpu device')
parser.add_argument('-mode', type=int, default=0, help='mode-0 : main task model substitute, mode-1 : pipeline substitute')
args = parser.parse_args()


enable_cuda = args.enable_cuda
gpu_id = args.gpu_id
mode = args.mode

ckpt_dir = 'checkpoint'

if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)


if enable_cuda:
    device = torch.device('cuda:%d' % gpu_id)
else:
    device = torch.device('cpu')


lr_rate = 0.01
batch_size = 200
n_iters = 20000
n_class = 12


def test(data,model):
    correct = 0
    tot = 0
    X,GT = data.sequential_val_batch()
    while X is not None:

        X = X.to(device)
        GT = GT.to(device)
        Y = model(X)
        Y = torch.argmax(Y,dim=1)

        this_batch_size = len(Y)
        
        for i in range(this_batch_size):
            tot+=1
            if GT[i] == Y[i]:
                correct+=1
        
        X,GT = data.sequential_val_batch()
    
    print('acc = %d / %d' % (correct,tot))
    return correct/tot



print('[Data] Preparing .... ')
data = DataSubstitute(mode=mode,batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
model = NEURAL(n_class=n_class, n_channel=3)
model = model.to(device)
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

    Y = model(X)
    loss = loss_f(Y,GT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    now = (time.time() - st) / 60.0

    if (i+1) % 1000 == 0 :
        print(' ### Eval ###')
        print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))
        model.eval()
        
        score = test(data,model)

        if score>max_acc:
            print('[save]..')
            max_acc = score
            stable_iter = 0
            torch.save(model.state_dict(), './checkpoint/model_' + str(save_n) + '.ckpt')
            save_n+=1
        else:
            stable_iter += 1
            if stable_iter == 10:
                print('Stable ... Training END ..')
                break
        model.train()
        #print('recall = %f, precision = %f' % (recall,precision))


print('[Training] Done ...')