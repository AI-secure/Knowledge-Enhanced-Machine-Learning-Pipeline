import torch
from torch import nn

class NEURAL(nn.Module): # end-to-end neural classifier 

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
        scale_1 = x.view(x.size(0), -1) 

        ## Scale 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        scale_2 = x.view(x.size(0), -1) 

        ## Scale 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout(x)
        scale_3 = x.view(x.size(0), -1) 

        ###### FC ########################
        embedding = torch.cat( (scale_1,scale_2,scale_3), dim = 1)
        
        x = self.fc1(embedding)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
