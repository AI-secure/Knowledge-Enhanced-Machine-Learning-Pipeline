import os
import time

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataset import DigitData
from model import Model

batch_size = 100
initial_learning_rate = 0.01
initial_patience = 100
decay_steps = 10000
decay_rate = 0.9
path_to_log_dir = './digit_log'
path_to_restore_checkpoint_file = None

def _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits, labels):
    length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, labels[:,0])
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, labels[:,1])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, labels[:,2])
    digit3_cross_entropy = torch.nn.functional.cross_entropy(digit3_logits, labels[:,3])
    digit4_cross_entropy = torch.nn.functional.cross_entropy(digit4_logits, labels[:,4])
    digit5_cross_entropy = torch.nn.functional.cross_entropy(digit5_logits, labels[:,5])
    loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    return loss

def _test(model,data):
    correct = 0
    tot = 0
    X,GT = data.sequential_val_batch()
    while X is not None:
        X = X.cuda()

        length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(X)
        length_prediction = length_logits.max(1)[1]
        digit1_prediction = digit1_logits.max(1)[1]
        digit2_prediction = digit2_logits.max(1)[1]
        digit3_prediction = digit3_logits.max(1)[1]
        digit4_prediction = digit4_logits.max(1)[1]
        digit5_prediction = digit5_logits.max(1)[1]

        this_batch_size = len(GT)

        for i in range(this_batch_size):
            tot+=1
            if GT[i][0] != length_prediction[i]: continue
            mark = True
            for k in range(1,6):
                if GT[i][k] != eval('digit%d_prediction' % k)[i] :
                    mark = False
            if mark :
                correct+=1

        X,GT = data.sequential_val_batch()

    #print('acc = %d / %d' % (correct,tot))
    return correct/tot
    


def _train():
    
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    step = 0
    patience = initial_patience
    best_accuracy = 0.0
    duration = 0.0

    print('preparing model ...')
    model = Model()
    #print('--- init from the SVHN model ---')
    #model.load_state_dict(torch.load('init.pth'),strict=False)
    model.cuda()
    print('done.')

    print('preparing data ...')
    data = DigitData(batch_size=batch_size)
    #print('=> Evaluating init performance...')
    #model.eval()
    #accuracy = _test(model,data)
    #model.train()
    #print('==> accuracy = %f' % (accuracy))
    print('done.')
    
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)

    if path_to_restore_checkpoint_file is not None:
        assert os.path.isfile(path_to_restore_checkpoint_file), '%s not found' % path_to_restore_checkpoint_file
        step = model.restore(path_to_restore_checkpoint_file)
        scheduler.last_epoch = step
        print('Model restored from file: %s' % path_to_restore_checkpoint_file)

    path_to_losses_npy_file = os.path.join(path_to_log_dir, 'losses.npy')
    if os.path.isfile(path_to_losses_npy_file):
        losses = np.load(path_to_losses_npy_file)
    else:
        losses = np.empty([0], dtype=np.float32)


    start_time = time.time()

    model.train()

    while True:

        features_batch, labels_batch = data.random_train_batch()
        features_batch = features_batch.cuda()
        labels_batch = labels_batch.cuda()

        length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(features_batch)
        loss = _loss(length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits, labels_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step % num_steps_to_show_loss == 0:
            duration = 0.0
            print('=> step %d, loss = %f, learning_rate = %f (Taken time : %f minutes)' % 
            (step, loss.item(), scheduler.get_lr()[0], (time.time()-start_time)/60.0))
        
        if step % num_steps_to_check != 0: continue

        losses = np.append(losses, loss.item())
        np.save(path_to_losses_npy_file, losses)

        print('=> Evaluating on validation dataset...')
        model.eval()
        accuracy = _test(model,data)
        model.train()
        print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

        if accuracy > best_accuracy:
            path_to_checkpoint_file = model.store(path_to_log_dir, step=step)
            print('=> Model saved to file: %s' % path_to_checkpoint_file)
            patience = initial_patience
            best_accuracy = accuracy
        else:
            patience -= 1

        print('=> patience = %d' % patience)
        if patience == 0:
            return



if not os.path.exists(path_to_log_dir):
    os.makedirs(path_to_log_dir)

print('Start training')
_train()
print('Done')