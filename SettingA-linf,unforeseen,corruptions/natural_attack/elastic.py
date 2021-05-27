import math
import numbers

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import random

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


# Taken from: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

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
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, inp):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(inp, weight=self.weight, groups=self.groups)

class ElasticDeformation(nn.Module):
    def __init__(self, im_size, filter_size, std):
        super().__init__()
        self.im_size = im_size
        self.filter_size = filter_size
        self.std = std
        self.kernel = GaussianSmoothing(2, self.filter_size, self.std).cuda()

        self._get_base_flow()

    def _get_base_flow(self):
        xflow, yflow = np.meshgrid(
                np.linspace(-1, 1, self.im_size, dtype='float32'),
                np.linspace(-1, 1, self.im_size, dtype='float32'))
        flow = np.stack((xflow, yflow), axis=-1)
        flow = np.expand_dims(flow, axis=0)
        self.base_flow = nn.Parameter(torch.from_numpy(flow)).cuda().detach()

    def warp(self, im, flow):
        return F.grid_sample(im, flow, mode='bilinear')

    def forward(self, im, params):
        flow = F.pad(params, ((self.filter_size - 1) // 2, ) * 4 , mode='reflect')
        local_flow = self.kernel(flow).transpose(1, 3).transpose(1, 2)
        return self.warp(im, local_flow + self.base_flow)


class ElasticAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol,
                 rand_init=True, scale_each=False,
                 kernel_size=25, kernel_std=3):
        '''
        Arguments:
            nb_its (int):          Number of iterations
            eps_max (float):       Maximum flow, in L_inf norm, in pixels
            step_size (float):     Maximum step size in L_inf norm, in pixels
            resol (int):           Side length of images, in pixels
            rand_init (bool):      Whether to do a random init
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            kernel_size (int):     Size, in pixels of gaussian kernel
            kernel_std (int):      Standard deviation of kernel
        '''
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.deformer = ElasticDeformation(resol, kernel_size, kernel_std)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, eps):
        if self.rand_init:
            # initialized randomly in [-1, 1], then scaled to [-base_eps, base_eps]
            flow = torch.rand((batch_size, 2, self.resol, self.resol),
                              dtype=torch.float32, device='cuda') * 2 - 1
            flow = eps[:, None, None, None] * flow
        else:
            flow = torch.zeros((batch_size, 2, self.resol, self.resol),
                               dtype=torch.float32, device='cuda')
        flow.requires_grad_()
        return flow
        
    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=True):
        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True

        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand * self.eps_max
            step_size = rand * self.step_size
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        # Our base_eps and step_size are in pixel scale, but flow is in [-1, 1] scale
        base_eps.mul_(2.0 / self.resol)
        step_size.mul_(2.0 / self.resol)

        flow = self._init(pixel_img.size()[0], base_eps)        
        pixel_inp_adv = self.deformer(pixel_inp, flow)

        if self.nb_its > 0:
            res = pixel_model(pixel_inp_adv)       
            for it in range(self.nb_its):
                loss = self.criterion(res, target)
                loss.backward()

                if avoid_target:
                    grad = flow.grad.data
                else:
                    grad = -flow.grad.data
                
                # step_size has already been converted to [-1, 1] scale
                flow.data = flow.data + step_size[:, None, None, None] * grad.sign()
                flow.data = torch.max(torch.min(flow.data, base_eps[:, None, None, None]), -base_eps[:, None, None, None])            
                pixel_inp_adv = self.deformer(pixel_inp, flow)
                if it != self.nb_its - 1:
                    res = pixel_model(pixel_inp_adv)
                    flow.grad.data.zero_()
        return pixel_inp_adv
