import torch 
from model import NEURAL
from dataset import DataMain
import time
import argparse
import os


device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


batch_size = 200
ckpt_root = './adv_checkpoint/'



def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40) :
    # let's do untargeted attack
    images = images.clone().to(device)
    labels = labels.to(device)
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=-0.5, max=0.5).detach_()

    return images

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


def test(data,model,attack_eps):
    
    correct_adv = 0
    correct = 0
    tot = 0
    X,GT = data.sequential_test_batch()

    while X is not None:

        X = X.to(device)
        X_adv = pgd_attack_random(model,X,GT,eps=attack_eps/255.0,alpha=1/255,iters=100,randomize=True)

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
        
        X,GT = data.sequential_test_batch()
    
    print('acc = %d/%d, adv_acc = %d/%d' % (correct,tot, correct_adv,tot))





print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
model = NEURAL(n_class=12,n_channel=3) 
model = model.to(device)
model.eval()
print('[Model] Done .... ')

#ckpt = torch.load(main_sensor_path,map_location = 'cpu')
#model.load_state_dict(ckpt)


for adv_model in [4,8,16,32]:
    ckpt = torch.load( ckpt_root + 'adv_%d.ckpt' % adv_model , map_location = device)
    model.load_state_dict(ckpt)
    print('------------- Adv training eps = %d/255 ---------------' % adv_model )
    for attack_eps in [0,2,4,8,16]:
        print('> attack_eps = %d/255' % attack_eps)
        test(data,model,attack_eps)
