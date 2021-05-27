import torch 
from model import NEURAL
from dataset import DataMain
import time

device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
lr_rate = 0.01
batch_size = 200
n_iters = 50000


def test(data,model):

    STOP_true_positive = 0
    STOP_false_positive = 0
    STOP_num = 0
    
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

            if GT[i] == 0:
                STOP_num+=1
                if Y[i]==0:
                    STOP_true_positive+=1
            elif Y[i]==0:
                STOP_false_positive+=1
            
        X,GT = data.sequential_val_batch()
    print('acc = %d / %d' % (correct,tot))


    recall = STOP_true_positive/STOP_num
    if STOP_true_positive+STOP_false_positive == 0:
        precision = 0
    else:
        precision = STOP_true_positive/(STOP_true_positive+STOP_false_positive)
    print('stop sign : recall = %d/%d = %f, precision = %d/%d = %f' % \
        (STOP_true_positive,STOP_num, recall, \
            STOP_true_positive,STOP_true_positive+STOP_false_positive, precision) )
    
    return correct/tot





print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
model = NEURAL(n_class=12,n_channel=3) 
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
    #if (i+1) % 10 == 0 :
        print(' ### Eval ###')
        print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))
        model.eval()
        
        score = test(data,model)

        if score>max_acc:
            print('[save]..')
            max_acc = score
            stable_iter = 0
            torch.save(model.state_dict(), './checkpoint/model_' + str(save_n) + '_acc=%f.ckpt'%(score))
            save_n+=1
        else:
            stable_iter += 1
            if stable_iter == 10:
                print('Stable ... Training END ..')
                break
        model.train()


print('[Training] Done ...')