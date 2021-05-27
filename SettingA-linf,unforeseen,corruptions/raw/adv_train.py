import torch 
from model import NEURAL
from dataset import DataMain
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-gpu_id', type=int, required=True, help='id number of the gpu device')
parser.add_argument('-attack_eps', type=int, required=True, help='noise magnitude in adversarial training')
args = parser.parse_args()

gpu_id = args.gpu_id
attack_eps = args.attack_eps

device = torch.device('cuda:%d' % gpu_id  if torch.cuda.is_available() else 'cpu')

lr_rate = 0.01
batch_size = 200
n_iters = 50000
main_sensor_path = '../pipeline/sensor/main/main.ckpt'

ckpt_root = 'adv_train_%d_ckpt/' % attack_eps
if not os.path.exists(ckpt_root):
    os.mkdir(ckpt_root)


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
    
    images = images.clone().to(device)
    for t in range(iters):
        loss = torch.nn.CrossEntropyLoss()(model(images + delta ), labels)
        loss.backward()
        
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-eps,eps)
        delta.data = (delta.data + images ).clamp(-0.5,0.5)-(images)
        delta.grad.zero_()
    
    return (delta+images).detach()

def test(data,model):
    
    correct_adv = 0
    correct = 0
    tot = 0
    X,GT = data.sequential_val_batch()
    while X is not None:

        X = X.to(device)
        X_adv = pgd_attack_random(model, X, GT, eps=attack_eps/255.0, alpha=1/255, iters=40, randomize=True)
        #pgd_attack(model,X,GT,eps=attack_eps/255.0,alpha=1/255,iters=40)

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
model = NEURAL(n_class=12,n_channel=3) 
ckpt = torch.load(main_sensor_path,map_location = 'cpu')
model.load_state_dict(ckpt)
model = model.to(device)
model.eval()
adv_acc = test(data,model)
print('[Load successfully] initial adv accuracy = %f' % adv_acc)
print('[Model] Done .... ')


if attack_eps >= 16:
    lr_rate = 0.002
    batch_size = 200


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

    # 20 iterations for adv-train
    X_adv = pgd_attack_random(model, X, GT, eps=attack_eps/255.0, alpha=2/255, iters=20, randomize=True)

    Y = model(X_adv)
    loss = loss_f(Y,GT)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    now = (time.time() - st) / 60.0

    if (i+1) % 10 ==0:
        print('[process: %d/%d] Loss = %f' % (i+1,n_iters,loss))

    if (i+1) % 1000 == 0 :
        print(' ### Eval ###')
        print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))
        model.eval()
        
        score = test(data,model)

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