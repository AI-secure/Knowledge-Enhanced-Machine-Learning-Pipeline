import random
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sparse
import torch.nn.functional as F

class Pixel2Real(nn.Module):
    def __init__(self, resol):
        super().__init__()
        
    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = (x / 255.) - 0.5
        return x

class Real2Pixel(nn.Module):
    def __init__(self, resol):
        super().__init__()

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [-0.5, 0.5]
        '''
        x = (x + 0.5) * 255.
        return x


class PixelModel(nn.Module):
    def __init__(self, model, resol):
        super().__init__()
        self.model = model
        self.transform = Pixel2Real(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to [0, 255]
        '''
        x = self.transform(x)
        # x is now normalized as the model expects
        x = self.model(x)
        return x

class AttackWrapper(nn.Module): 

    # deal with normzalized img ~ [-0.5,0.5]

    def __init__(self, resol):
        super().__init__()
        self.resol = resol
        self.transform = Pixel2Real(resol)
        self.inverse_transform = Real2Pixel(resol)
        self.epoch = 0
        
    def forward(self, model, img, *args, **kwargs):
        # img : [-0.5,0.5]
        # model : raw model
        # pixel_model : wrapped model, so that the input is pixel-space scale, there would be an inner transform

        was_training = model.training
        pixel_model = PixelModel(model, self.resol)
        pixel_model.eval()

        pixel_img = self.inverse_transform(img.detach())

        pixel_ret = self._forward(pixel_model, pixel_img, *args, **kwargs)
        if was_training:
            pixel_model.train()
        ret = self.transform(pixel_ret)
        return ret

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._update_params(epoch)

    def _update_params(self, epoch):
        pass

def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True

def normalize(orig):
    batch_size = orig.size(0)
    omax = torch.max(orig.view(batch_size, -1), 1)[0].detach().view(batch_size,1,1,1)
    omin  = torch.min(orig.view(batch_size, -1), 1)[0].detach().view(batch_size,1,1,1)
    return (orig - omin) / (omax - omin)

def get_gabor(k_size, sigma, Lambda, theta):
    y, x = torch.meshgrid([torch.linspace(-0.5, 0.5, k_size), torch.linspace(-0.5, 0.5, k_size)])
    rotx = x * torch.cos(theta) + y * torch.sin(theta)
    roty = -x * torch.sin(theta) + y * torch.cos(theta)
    g = torch.zeros(y.shape)
    g = torch.exp(-0.5 * (rotx ** 2 / (sigma + 1e-3) ** 2 + roty ** 2 / (sigma + 1e-3) ** 2))
    g = g * torch.cos(2 * np.pi * Lambda * rotx)
    return g

def get_gabor_with_sides(k_size, sigma, Lambda, theta, sides=3):
    g = get_gabor(k_size, sigma, Lambda, theta)
    for i in range(1, sides):
        g += get_gabor(k_size, sigma, Lambda, theta + np.pi * i / sides)
    return g

def normalize_var(orig):
    batch_size = orig.size(0)

    # Spectral variance
    mean = torch.mean(orig.view(batch_size, -1), 1).view(batch_size, 1, 1, 1)
    spec_var = torch.rfft(torch.pow(orig -  mean, 2), 2)

    # Normalization
    imC = torch.sqrt(torch.irfft(spec_var, 2, signal_sizes=orig.size()[2:]).abs())
    imC /= torch.max(imC.view(batch_size, -1), 1)[0].view(batch_size, 1, 1, 1)
    minC = 0.001
    imK =  (minC + 1) / (minC + imC)

    mean, imK = mean.detach(), imK.detach()
    img = mean + (orig -  mean) * imK
    return normalize(img)

def gabor_rand_distributed(sp_conv, gabor_kernel):
    # code converted from https://github.com/kenny-co/procedural-advml

    batch_size = sp_conv.size(0)
    # reshape batch dimension to channel dimension to use group convolution
    # so that data processes in parallel
    sp_conv = sp_conv.view(1, batch_size, sp_conv.size(-2), sp_conv.size(-1))
    sp_conv = F.conv2d(sp_conv, weight=gabor_kernel, stride=1, groups=batch_size, padding=11)
    sp_conv = sp_conv.view(batch_size, 1, sp_conv.size(-2), sp_conv.size(-1))

    return normalize_var(sp_conv)

class GaborAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, num_kern):
        grid = 14

        if self.rand_init:
            sparse_matrices = []
            sp_conv_numpy = sparse.random(self.resol*batch_size, self.resol,
                            density= 1. / grid, format='csr')
            sp_conv_numpy.data = sp_conv_numpy.data * 2 - 1
            sp_conv = torch.FloatTensor(sp_conv_numpy.todense()).view(
                        batch_size, self.resol, self.resol)

            mask = (sp_conv == 0).cuda().float().view(-1, 1, self.resol, self.resol)
            gabor_vars = sp_conv.clone().cuda().view(-1, 1, self.resol, self.resol)
            gabor_vars.requires_grad_(True)
        return gabor_vars, mask

    def _get_gabor_kernel(self, batch_size):
        # make gabor filters to convolve with variables
        k_size = 23
        kernels = []
        for b in range(batch_size):
            sides = np.random.randint(10) + 1
            sigma = 0.3 * torch.rand(1) +  0.1
            Lambda = (k_size / 4. - 3) * torch.rand(1) + 3
            theta = np.pi * torch.rand(1)

            kernels.append(get_gabor_with_sides(k_size, sigma, Lambda, theta, sides).cuda())
        gabor_kernel = torch.cat(kernels, 0).view(-1, 1, k_size, k_size)
        return gabor_kernel

    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        pixel_inp = pixel_img.detach()
        batch_size = pixel_img.size(0)

        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand.mul(self.eps_max)
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        gabor_kernel = self._get_gabor_kernel(batch_size)
        num_kern = np.random.randint(50) + 1
        gabor_vars, mask = self._init(batch_size, num_kern)
        gabor_noise = gabor_rand_distributed(gabor_vars, gabor_kernel)
        gabor_noise = gabor_noise.expand(-1, 3, -1, -1)
        s = pixel_model(torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_noise, 0., 255.))
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = gabor_vars.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -gabor_vars.grad.data

            grad_sign = grad.sign()
            gabor_vars.data = gabor_vars.data + step_size[:, None, None, None] * grad_sign
            gabor_vars.data = torch.clamp(gabor_vars.data, -1, 1) * mask

            if it != self.nb_its - 1:
                gabor_noise = gabor_rand_distributed(gabor_vars, gabor_kernel).expand(-1, 3, -1, -1)
                s = pixel_model(torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_noise, 0., 255.))
                gabor_vars.grad.data.zero_()
        pixel_result = torch.clamp(pixel_inp + base_eps[:, None, None, None] * gabor_rand_distributed(gabor_vars, gabor_kernel), 0., 255.)
        return pixel_result
