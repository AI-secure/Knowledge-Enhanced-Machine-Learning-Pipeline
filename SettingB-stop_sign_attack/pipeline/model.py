import glob
import os

import torch
import torch.jit
import torch.nn as nn


class Model(torch.jit.ScriptModule): # digit recognition model
    CHECKPOINT_FILENAME_PATTERN = 'model-{}.pth'

    def __init__(self):
        super(Model, self).__init__()

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self._hidden9 = nn.Sequential(
            nn.Linear(192 * 7 * 7, 3072),
            nn.ReLU()
        )
        self._hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._digit_length = nn.Sequential(nn.Linear(3072, 7)) # 0~5 + violation
        self._digit1 = nn.Sequential(nn.Linear(3072, 11)) # 0~9 + no
        self._digit2 = nn.Sequential(nn.Linear(3072, 11))
        self._digit3 = nn.Sequential(nn.Linear(3072, 11))
        self._digit4 = nn.Sequential(nn.Linear(3072, 11))
        self._digit5 = nn.Sequential(nn.Linear(3072, 11))

    @torch.jit.script_method
    def forward(self, x):
        x = self._hidden1(x)
        x = self._hidden2(x)
        x = self._hidden3(x)
        x = self._hidden4(x)
        x = self._hidden5(x)
        x = self._hidden6(x)
        x = self._hidden7(x)
        x = self._hidden8(x)
        x = x.reshape(x.size(0), 192 * 7 * 7)
        x = self._hidden9(x)
        x = self._hidden10(x)

        length_logits = self._digit_length(x)
        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)
        digit3_logits = self._digit3(x)
        digit4_logits = self._digit4(x)
        digit5_logits = self._digit5(x)

        return length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits

    def store(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(path_to_model.split('/')[-1][6:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(path_to_dir, Model.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file, resume_step = False):
        self.load_state_dict(torch.load(path_to_checkpoint_file),strict=False)
        if resume_step:
            step = int(path_to_checkpoint_file.split('/')[-1][6:-4])
            return step
        else:
            return None



class NEURAL(nn.Module): # GTSRB-CNN

    def __init__(self,n_class,n_channel):

        super(NEURAL,self).__init__()

        ########################### Learn a color transform ###########################
        self.conv0 = nn.Sequential(                     
            nn.Conv2d(n_channel,3,1),       #input_size=(n_channel*32*32)
            nn.ReLU()
        ) # output_size=(3*32*32)

        ########################### Level-1 ###########################
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),   #input_size=(3*32*32)
            nn.ReLU()
        ) # output_size=(32*32*32)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32,5,1,2),   #input_size=(32*32*32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        ) # output_size=(32*16*16)

        ########################### Level-2 ###########################
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),   #input_size=(32*16*16)
            nn.ReLU()
        ) # output_size=(64*16*16)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),   #input_size=(64*16*16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        ) # output_size=(64*8*8)

        ########################### Level-3 ###########################
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),   #input_size=(64*8*8)
            nn.ReLU()
        ) # output_size=(128*8*8)

        self.conv6 = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),   #input_size=(128*8*8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        ) # output_size=(128*4*4)

        
        # num_fc = 32*16*16 + 64*8*8 + 128*4*4 = 14336
        self.fc1 = nn.Sequential(
                    nn.Linear(14336,1024),
                    nn.ReLU()
                )
        self.fc2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1024,n_class)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self,x):
        x = self.conv0(x) # Color Transform

        ## Scale 1 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        scale_1 = x.reshape(x.size(0), -1) 

        ## Scale 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        scale_2 = x.reshape(x.size(0), -1) 

        ## Scale 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout(x)
        scale_3 = x.reshape(x.size(0), -1) 

        ###### FC ########################
        embedding = torch.cat( (scale_1,scale_2,scale_3), axis = 1)
        
        x = self.fc1(embedding)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
