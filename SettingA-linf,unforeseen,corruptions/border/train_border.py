import torch
from model import NEURAL
from dataset import BorderData
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-gpu_id', type=int, required=True, help='id number of the gpu device')
parser.add_argument('-pid', type=int, required=True, help='id number of the border type to be detected.')
parser.add_argument('-isconservative', type=int, required=True, help='conservative or not?')
args = parser.parse_args()

gpu_id = args.gpu_id
pid = args.pid
isconservative = args.isconservative

ckpt_dir = 'ckpt_pid=%d_isconservative=%d' % (pid,isconservative)

if isconservative:
    w_p = 1
    w_n = 20
    thr_TPR = 0.5
    thr_TNR = 0.95
    if pid == 7:
        w_n = 5
else:
    w_p = 20
    w_n = 1
    thr_TPR = 0.95
    thr_TNR = 0.5

    if pid == 5:
        thr_TPR = 0.85
        #w_p = 100
        w_p = 60
        w_n = 0.5
    
    if pid == 7:
        #w_p = 50
        w_p = 30
        w_n = 0.5
    
    if pid == 0 :
        w_p = 50
        w_n = 0.5
    

print('w_p = %f , w_n = %f' % (w_p,w_n))


if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
print('ckpt_dir : ',ckpt_dir)
print('thr_TPR=%f, thr_TNR = %f' % (thr_TPR, thr_TNR))
print('w_p=%f, w_n = %f' % (w_p, w_n))

device = torch.device('cuda:%d' % gpu_id  if torch.cuda.is_available() else 'cpu')
print('working device : ', device)


lr_rate = 0.001
batch_size = 200
n_iters = 40000


def test(data,model):

    TPR = 0
    positive_num = 0

    TNR = 0
    negative_num = 0

    X = data.sequential_test_positive_batch()
    while X is not None:
        X = X.to(device)
        Y = model(X)
        Y = torch.argmax(Y,dim=1)
        positive_num += len(Y)
        for item in Y:
            if item==1:
                TPR+=1
        X = data.sequential_test_positive_batch()



    X = data.sequential_test_negative_batch()
    while X is not None:
        X = X.to(device)
        Y = model(X)
        Y = torch.argmax(Y,dim=1)
        negative_num += len(Y)
        for item in Y:
            if item==0:
                TNR+=1
        X = data.sequential_test_negative_batch()

    # return recall, precision
    print('TPR : %d / %d' % (TPR,positive_num))
    print('TNR : %d / %d' % (TNR,negative_num))

    return TPR/positive_num , TNR/negative_num






print('[Data] Preparing .... ')
data = BorderData(batch_size=batch_size,pid=pid)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
model = NEURAL(n_class=2,n_channel=3)
model = model.to(device)
print('[Model] Done .... ')




loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay = 1e-4)
st = time.time()

GT_positive = torch.ones(batch_size).long().to(device)
GT_negative = torch.zeros(batch_size).long().to(device)

model.train()

n_save = 0

for i in range(n_iters):

    positive_samples, negative_samples = data.random_train_batch()
    positive_samples = positive_samples.to(device)
    negative_samples = negative_samples.to(device)

    Y_p = model(positive_samples)
    loss_p = loss_f(Y_p, GT_positive)

    Y_n = model(negative_samples)
    loss_n = loss_f(Y_n,GT_negative)

    loss = loss_p*w_p + loss_n*w_n

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    now = (time.time() - st) / 60.0
    if (i+1) % 5000 == 0 :
        print(' ### Eval ###')
        print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))

        model.eval()
        TPR, TNR = test(data,model)


        if TPR > thr_TPR and TNR > thr_TNR:
            ckpt_name = 'model_%d_TPR=%f_TNR=%f.ckpt' % (n_save,TPR,TNR)
            torch.save(model.state_dict(), ckpt_dir+'/'+ckpt_name)
            print('[save] : ',ckpt_dir+'/'+ckpt_name)
            n_save+=1

        model.train()


print('[Training] Done ...')