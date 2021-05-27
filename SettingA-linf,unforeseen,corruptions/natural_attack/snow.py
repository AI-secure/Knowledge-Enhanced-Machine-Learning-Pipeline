import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
import numpy as np


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

# modification of code from pytorch forums
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size, sigma, channels=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * 2
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * 2

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = F.conv2d

    def forward(self, input, padding=0):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=padding)


# from stackoverflow
def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def make_kernels(snow_length_bound=13, blur=True):
    kernels = []

    flip = np.random.uniform() < 0.5

    for i in range(7):
        k_size = snow_length_bound
        mid = k_size//2
        k_npy = np.zeros((k_size, k_size))
        rr, cc, val = weighted_line(
            mid, mid, np.random.randint(mid+2,k_size), np.random.randint(mid+2,k_size),
            np.random.choice([1,3,5], p=[0.6, 0.3, 0.1]), mid, k_size)

        k_npy[rr, cc] = val
        k_npy[:mid+1, :mid+1] = k_npy[::-1,::-1][:mid+1,:mid+1]

        if flip:
            k_npy = k_npy[:, ::-1]

        kernel = torch.FloatTensor(k_npy.copy()).view(1,1,k_size,k_size).cuda()

        if blur:
            blurriness = np.random.uniform(0.41, 0.6)
            gaussian_blur = GaussianSmoothing(int(np.ceil(5 * blurriness)), blurriness)
            kernel = gaussian_blur(kernel, padding=1)
        kernels.append(kernel)

    return kernels


def snow_creator(intensities, k, resol):
    flake_grids = []
    k = torch.cat(k, 1)

    intensities_pow = torch.pow(intensities, 4)
    flake_grids = torch.zeros((intensities.size(0), k.size(1), resol, resol)).cuda()

    for i in range(4):
        flake_grids[:, i, ::4,i::4] = intensities_pow[:,i]
    for i in range(3):
        flake_grids[:, i+4, i+1::4,::4] = intensities_pow[:,4+i]

    snow = F.conv2d(flake_grids, k, padding=k.size(-1)//2)

    return snow



def apply_snow(img, snow, scale, discolor=0.25):    
    out = (1 - discolor) * img +\
    discolor * torch.max(img, (0.2126 * img[:,0:1] + 0.7152 * img[:,1:2] + 0.0722 * img[:,2:3]) * 1.5 + 0.5)   
    return torch.clamp(out + scale[:, None, None, None] * snow, 0, 1)

class SnowAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False,
                 budget=0.2):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            budget (float):        Controls rate parameter of snowflakes
        """
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.budget = budget

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its    

    def _init(self, batch_size):
        # flake intensities follow an exponential distribution
        flake_intensities = torch.exp(-1./(self.budget)*torch.rand(batch_size,7,self.resol//4,self.resol//4)).cuda()
        flake_intensities.requires_grad_(True)
        
        return flake_intensities

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
        
        flake_intensities = self._init(batch_size)
        kernels = make_kernels()
        snow = snow_creator(flake_intensities, kernels, self.resol)
        s = pixel_model(apply_snow(pixel_inp / 255., snow, base_eps) * 255)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''            
            if avoid_target:
                # to avoid the target, we increase the loss
                grad = flake_intensities.grad.data
            else:
                # to hit the target, we reduce the loss
                grad = -flake_intensities.grad.data

            grad_sign = grad.sign()
            flake_intensities.data = flake_intensities.data + step_size[:, None, None, None] * grad_sign

            if it != self.nb_its - 1:
                snow = snow_creator(flake_intensities, kernels, self.resol)
                s = pixel_model(apply_snow(pixel_inp / 255., snow, base_eps) * 255)
                flake_intensities.grad.data.zero_()

                # keep intensities in valid range
                flake_intensities.detach()
                flake_intensities.data = flake_intensities.data.clamp(1e-9, 1)

                # ensure intensities do not exceed budget
                block_size = 8
                blocks = flake_intensities.size(-1)//block_size

                budget_per_region = F.adaptive_avg_pool2d(flake_intensities, blocks)
                budget_per_region[budget_per_region < self.budget] = self.budget

                for i in range(blocks):
                    for j in range(blocks):
                        flake_intensities.data[
                            :,:,i*block_size:(i+1)*block_size,
                            j*block_size:(j+1)*block_size] *= self.budget/budget_per_region[:,:,i,j].view(-1,7,1,1)

                flake_intensities.requires_grad_()

        snow = snow_creator(flake_intensities, kernels, self.resol)
        pixel_result = apply_snow(pixel_inp / 255., snow, base_eps) * 255
        return pixel_result

