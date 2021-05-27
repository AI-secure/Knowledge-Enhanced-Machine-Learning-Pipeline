import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import itertools




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






# From https://gist.github.com/deanmark/9aec75b7dc9fa71c93c4bc85c5438777
def tensordot_pytorch(a, b, dims=2):
    axes=dims
    # code adapted from numpy
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1
    
    # uncomment in pytorch >= 0.5
    # a, b = torch.as_tensor(a), torch.as_tensor(b)
    as_ = a.shape
    nda = a.dim()
    bs = b.shape
    ndb = b.dim()
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.permute(newaxes_a).reshape(newshape_a)
    bt = b.permute(newaxes_b).reshape(newshape_b)

    res = at.matmul(bt)
    return res.reshape(olda + oldb)

# based on https://github.com/rshin/differentiable-jpeg/blob/master/jpeg.py
# Thanks to Dan Hendrycks for adapting the code to pytorch

def rgb_to_ycbcr(image):
  matrix = np.array(
      [[65.481, 128.553, 24.966],
       [-37.797, -74.203, 112.],
       [112., -93.786, -18.214]],
      dtype=np.float32).T / 255
  shift = torch.as_tensor([16., 128., 128.], device="cuda")

  # result = torch.tensordot(image, torch.as_tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, matrix, dims=1) + shift
  result.view(image.size())
  return result


def rgb_to_ycbcr_jpeg(image):
  matrix = np.array(
      [[0.299, 0.587, 0.114],
       [-0.168736, -0.331264, 0.5],
       [0.5, -0.418688, -0.081312]],
      dtype=np.float32).T
  shift = torch.as_tensor([0., 128., 128.], device="cuda")

  # result = torch.tensordot(image, torch.as_tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, torch.as_tensor(matrix, device='cuda'), dims=1) + shift
  result.view(image.size())
  return result


# 2. Chroma subsampling
def downsampling_420(image):
  # input: batch x height x width x 3
  # output: tuple of length 3
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  y, cb, cr = image[...,0], image[...,1], image[...,2]
  # requires that height and width are divisible by 2 to avoid
  # padding issues
  cb = F.avg_pool2d(cb, kernel_size=2)
  cr = F.avg_pool2d(cr, kernel_size=2)
  return y, cb, cr
  # return (y.squeeze(), cb.squeeze(), cr.squeeze())


# 3. Block splitting
# From https://stackoverflow.com/questions/41564321/split-image-tensor-into-small-patches
def image_to_patches(image):
  # input: batch x h x w
  # output: batch x h*w/64 x h x w
  k = 8
  batch_size, height, width = image.size()
  image_reshaped = image.view(batch_size, height // k, k, -1, k)
  image_transposed = torch.transpose(image_reshaped, 2, 3)
  return image_transposed.contiguous().view(batch_size, -1, k, k)


# 4. DCT
def dct_8x8_ref(image):
  image = image - 128
  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
          (2 * y + 1) * v * np.pi / 16)
    result[u, v] = value
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = np.outer(alpha, alpha) * 0.25
  return result * scale


def dct_8x8(image):
  image = image - 128
  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
        (2 * y + 1) * v * np.pi / 16)
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  scale = torch.FloatTensor(np.outer(alpha, alpha) * 0.25).cuda()
  #result = scale * torch.tensordot(image, torch.as_tensor(tensor, device="cuda"), dims=2)
  result = scale * tensordot_pytorch(image, torch.as_tensor(tensor, device="cuda"), dims=2)
  result.view(image.size())
  return result


def make_quantization_tables(self):
    # 5. Quantizaztion
    self.y_table = torch.as_tensor(np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]],
        dtype=np.float32).T, device="cuda")
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    self.c_table = torch.as_tensor(c_table, device="cuda")


def y_quantize(self, image, rounding, rounding_var, factor=1):
  image = image / (self.y_table * factor)
  image = rounding(image, rounding_var)
  return image


def c_quantize(self, image, rounding, rounding_var, factor=1):
  image = image / (self.c_table * factor)
  image = rounding(image, rounding_var)
  return image


# -5. Dequantization
def y_dequantize(self, image, factor=1):
  return image * (self.y_table * factor)


def c_dequantize(self, image, factor=1):
  return image * (self.c_table * factor)


# -4. Inverse DCT
def idct_8x8_ref(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = np.outer(alpha, alpha)
  image = image * alpha

  result = np.zeros((8, 8), dtype=np.float32)
  for u, v in itertools.product(range(8), range(8)):
    value = 0
    for x, y in itertools.product(range(8), range(8)):
      value += image[x, y] * np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
          (2 * v + 1) * y * np.pi / 16)
    result[u, v] = value
  return result * 0.25 + 128


def idct_8x8(image):
  alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
  alpha = torch.FloatTensor(np.outer(alpha, alpha)).cuda()
  image = image * alpha

  tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
  for x, y, u, v in itertools.product(range(8), repeat=4):
    tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
        (2 * v + 1) * y * np.pi / 16)
  # result = 0.25 * torch.tensordot(image, torch.as_tensor(tensor, device="cuda"), dims=2) + 128
  result = 0.25 * tensordot_pytorch(image, torch.as_tensor(tensor, device="cuda"), dims=2) + 128
  result.view(image.size())
  return result


# -3. Block joining
def patches_to_image(patches, height, width):
  # input: batch x h*w/64 x h x w
  # output: batch x h x w
  height = int(height)
  width = int(width)
  k = 8
  batch_size = patches.size(0)
  image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
  image_transposed = torch.transpose(image_reshaped, 2, 3)
  return image_transposed.contiguous().view(batch_size, height, width)

# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
  # input:
  #   y:  batch x height x width
  #   cb: batch x height/2 x width/2
  #   cr: batch x height/2 x width/2
  # output:
  #   image: batch x height x width x 3
  def repeat(x, k=2):
    height, width = x.size()[1:3]
    x = x.unsqueeze(-1)
    x = x.repeat((1, 1, k, k))
    x = x.view(-1, height * k, width * k)
    return x

  cb = repeat(cb)
  cr = repeat(cr)
  return torch.stack((y, cb, cr), dim=-1)


# -1. YCbCr -> RGB
def ycbcr_to_rgb(image):
  matrix = np.array(
      [[298.082, 0, 408.583],
       [298.082, -100.291, -208.120],
       [298.082, 516.412, 0]],
      dtype=np.float32).T / 256
  shift = torch.as_tensor([-222.921, 135.576, -276.836], device="cuda")

  # result = torch.tensordot(image, torch.tensor(matrix, device="cuda"), dims=1) + shift
  result = tensordot_pytorch(image, torch.tensor(matrix, device="cuda"), dims=1) + shift
  result.view(image.size())
  return result


def ycbcr_to_rgb_jpeg(image):
  matrix = np.array(
      [[1., 0., 1.402],
       [1, -0.344136, -0.714136],
       [1, 1.772, 0]],
      dtype=np.float32).T
  shift = torch.FloatTensor([0, -128, -128]).cuda()

  # result = torch.tensordot(image + shift, torch.tensor(matrix, device="cuda"), dims=1)
  result = tensordot_pytorch(image + shift, torch.tensor(matrix, device="cuda"), dims=1)
  result.view(image.size())
  return result


def jpeg_compress_decode(self, image_channels_first, rounding_vars, lambder, downsample_c=True,
                         factor=1):
  def noisy_round(x, noise):
    return x + lambder[:, None, None, None] * (noise - 0.5)
  
  #image = torch.as_tensor(image)
  image = torch.transpose(image_channels_first, 1, 3)
  height, width = image.size()[1:3]

  orig_height, orig_width = height, width
  if height % 16 != 0 or width % 16 != 0:
    # Round up to next multiple of 16
    height = ((height - 1) // 16 + 1) * 16
    width = ((width - 1) // 16 + 1) * 16

    vpad = height - orig_height
    wpad = width - orig_width
    top = vpad // 2
    bottom = vpad - top
    left = wpad // 2
    right = wpad - left

    # image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
    # image = tf.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'SYMMETRIC')
    # image = F.pad(image, (vpad, wpad), 'replicate')
    image = F.pad(image, (left, right, top, bottom), 'replicate')

  # "Compression"
  image = rgb_to_ycbcr_jpeg(image)
  if downsample_c:
    y, cb, cr = downsampling_420(image)
  else:
    y, cb, cr = torch.split(image, 3, dim=3)
  components = {'y': y, 'cb': cb, 'cr': cr}
  for k in components.keys():
    comp = components[k]
    comp = image_to_patches(comp)
    comp = dct_8x8(comp)
    if k == 'y':
        comp = y_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[0], factor)
    elif k  == 'cb':
        comp = c_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[1], factor)
    else:
        comp = c_quantize(self, comp, noisy_round, 0.5 + 0.5 * rounding_vars[2], factor)
    components[k] = comp

  # Decoding
  for k in components.keys():
    comp = components[k]
    comp = c_dequantize(self, comp, factor) if k in ('cb', 'cr') else y_dequantize(
        self, comp, factor)
    comp = idct_8x8(comp)
    if k in ('cb', 'cr'):
      if downsample_c:
        comp = patches_to_image(comp, height / 2, width / 2)
      else:
        comp = patches_to_image(comp, height, width)
    else:
      comp = patches_to_image(comp, height, width)
    components[k] = comp

  y, cb, cr = components['y'], components['cb'], components['cr']
  if downsample_c:
    image = upsampling_420(y, cb, cr)
  else:
    image = torch.stack((y, cb, cr), dim=-1)
  image = ycbcr_to_rgb_jpeg(image)

  # Crop to original size
  if orig_height != height or orig_width != width:
    #image = image[:, top:-bottom, left:-right]
    image = image[:, :-vpad, :-wpad]

  # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
  #    min_value = tf.minimum(tf.reduce_min(image), 0.)
  #    max_value = tf.maximum(tf.reduce_max(image), 255.)
  #    value_range = max_value - min_value
  #    image = 255 * (image - min_value) / value_range
  image = torch.clamp(image, 0, 255)

  return torch.transpose(image, 1, 3)

def quality_to_factor(quality):
    if quality < 50:
        return 50./quality
    else:
        return (200. - quality * 2)/100.


class JPEG(nn.Module):
    def __init__(self):
        super(JPEG, self).__init__()
        make_quantization_tables(self)

    def forward(self, pixel_inp, rounding_vars, epsilon):
        return jpeg_compress_decode(self, pixel_inp, rounding_vars, epsilon)
        





class JPEGAttack(AttackWrapper):
    def __init__(self, nb_its, eps_max, step_size, resol,
                 rand_init=True, opt='linf', scale_each=False, l1_max=2.):
        '''
        Arguments:
            nb_its (int):          Number of iterations
            eps_max (float):       Maximum flow, in L_inf norm, in pixels
            step_size (float):     Maximum step size in L_inf norm, in pixels
            resol (int):           Side length of images, in pixels
            rand_init (bool):      Whether to do a random init
            opt (string):          Which optimization algorithm to use, either 'linf', 'l1', or 'l2'
            scale_each (bool):     Whether to scale eps for each image in a batch separately
        '''
        super().__init__(resol)
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.rand_init = rand_init
        self.opt = opt
        print('JPEG_opt = %s' % self.opt)
        if opt not in ['linf', 'l1', 'l2']:
            raise NotImplementedError
        self.scale_each = scale_each
        self.l1_max = l1_max
        
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = nb_its
        self.jpeg = JPEG().cuda()

    def _convert_cat_var(self, cat_var, batch_size, height, width):
        y_var = cat_var[:, :height//8 * width//8 * 8 * 8].view((batch_size, height//8 * width//8, 8, 8))
        cb_var = cat_var[:, height//8 * width//8 * 8 * 8: height//8 * width//8 * 8 * 8 + height//16 * width//16 * 8 * 8].view((batch_size, height//16 * width//16, 8, 8))
        cr_var = cat_var[:, height//8 * width//8 * 8 * 8 + height//16 * width//16 * 8 * 8: height//8 * width//8 * 8 * 8 + 2 * height//16 * width//16 * 8 * 8].view((batch_size, height//16 * width//16, 8, 8))
        return y_var, cb_var, cr_var
    
    def _jpeg_cat(self, pixel_inp, cat_var, base_eps, batch_size, height, width):
        y_var, cb_var, cr_var = self._convert_cat_var(cat_var, batch_size, height, width)
        return self.jpeg(pixel_inp, [y_var, cb_var, cr_var], base_eps)
        
    def _run_one_pgd(self, pixel_model, pixel_inp, cat_var, target, base_eps, step_size, avoid_target=True):
        batch_size, channels, height, width = pixel_inp.size()
        pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var, base_eps, batch_size, height, width)
        s = pixel_model(pixel_inp_jpeg)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()

            if avoid_target:
                grad = cat_var.grad.data
            else:
                grad = -cat_var.grad.data

            if self.opt == 'linf':
                grad_sign = grad.sign()
                cat_var.data = cat_var.data + step_size[:, None] * grad_sign
                cat_var.data = torch.max(torch.min(cat_var.data, base_eps[:, None]),
                                         -base_eps[:, None]) # cannot use torch.clamp for tensors
            elif self.opt == 'l2':
                batch_size = pixel_inp.size()[0]
                grad_norm = torch.norm(grad.view(batch_size, -1), 2.0, dim=1)
                normalized_grad = grad / grad_norm[:, None]                
                cat_var.data = cat_var.data + step_size[:, None] * normalized_grad
                l2_delta = torch.norm(cat_var.data.view(batch_size, -1), 2.0, dim=1)
                proj_scale = torch.min(torch.ones_like(l2_delta, device='cuda'), base_eps / l2_delta)
                cat_var.data *= proj_scale[:, None]
                cat_var.data = torch.clamp(cat_var.data, -self.l1_max, self.l1_max)
                    
            if it != self.nb_its - 1:
                # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
                # its coordinates to [-1, 1]
                cat_var_temp = cat_var / base_eps[:, None]
                pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
                s = pixel_model(pixel_inp_jpeg)
            cat_var.grad.data.zero_()
        return cat_var
        
    def _run_one_fw(self, pixel_model, pixel_inp, cat_var, target, base_eps, avoid_target=True):
        batch_size, channels, height, width = pixel_inp.size()
        pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var, base_eps, batch_size, height, width)
        s = pixel_model(pixel_inp_jpeg)

        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()

            if avoid_target:
                grad = cat_var.grad.data
            else:
                grad = -cat_var.grad.data

            def where_float(cond, if_true, if_false):
                return cond.float() * if_true + (1-cond.float()) * if_false

            def where_long(cond, if_true, if_false):
                return cond.long() * if_true + (1-cond.long()) * if_false

            abs_grad = torch.abs(grad).view(batch_size, -1)
            num_pixels = abs_grad.size()[1]
            sign_grad = torch.sign(grad)

            bound = where_float(sign_grad > 0, self.l1_max - cat_var, cat_var + self.l1_max).view(batch_size, -1)
                
            k_min = torch.zeros((batch_size,1), dtype=torch.long, requires_grad=False, device='cuda')
            k_max = torch.ones((batch_size,1), dtype=torch.long, requires_grad=False, device='cuda') * num_pixels
                
            # cum_bnd[k] is meant to track the L1 norm we end up with if we take 
            # the k indices with the largest gradient magnitude and push them to their boundary values (0 or 255)
            values, indices = torch.sort(abs_grad, descending=True)
            bnd = torch.gather(bound, 1, indices)
            # subtract bnd because we don't want the cumsum to include the final element
            cum_bnd = torch.cumsum(bnd, 1) - bnd
                
            # this is hard-coded as floor(log_2(256 * 256 * 3))
            for _ in range(17):
                k_mid = (k_min + k_max) // 2
                l1norms = torch.gather(cum_bnd, 1, k_mid)
                k_min = where_long(l1norms > base_eps, k_min, k_mid)
                k_max = where_long(l1norms > base_eps, k_mid, k_max)
                
            # next want to set the gradient of indices[0:k_min] to their corresponding bound
            magnitudes = torch.zeros((batch_size, num_pixels), requires_grad=False, device='cuda')
            for bi in range(batch_size):
                magnitudes[bi, indices[bi, :k_min[bi,0]]] = bnd[bi, :k_min[bi,0]]
                magnitudes[bi, indices[bi, k_min[bi,0]]] = base_eps[bi] - cum_bnd[bi, k_min[bi,0]]
                
            delta_it = sign_grad * magnitudes.view(cat_var.size())
            # These should always be exactly epsilon
            # l1_check = torch.norm(delta_it.view(batch_size, -1), 1.0, dim=1) / num_pixels
            # print('l1_check: %s' % l1_check)
            cat_var.data = cat_var.data + (delta_it - cat_var.data) / (it + 1.0)

            if it != self.nb_its - 1:
                # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
                # its coordinates to [-1, 1]
                cat_var_temp = cat_var / base_eps[:, None]
                pixel_inp_jpeg = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
                s = pixel_model(pixel_inp_jpeg)
            cat_var.grad.data.zero_()
        return cat_var

    def _init_empty(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        return torch.zeros(shape, device='cuda')
    
    def _init_linf(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        return torch.rand(shape, device='cuda') * 2 - 1
        
    def _init_l1(self, batch_size, height, width):
        # returns random initialization with L_1 norm of 1
        # algorithm from https://arxiv.org/abs/math/0503650, Theorem 1
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        exp = torch.empty(shape, dtype=torch.float32, device='cuda')
        exp.exponential_()
        signs = torch.sign(torch.randn(shape, dtype=torch.float32, device='cuda'))
        exp = exp * signs
        exp_y = torch.empty(shape[0], dtype=torch.float32, device='cuda')
        exp_y.exponential_()
        norm = exp_y + torch.norm(exp.view(shape[0], -1), 1.0, dim=1)
        init = exp / norm[:, None]
        return init

    def _init_l2(self, batch_size, height, width):
        shape = (batch_size, (height//8 * width//8 + height//16 * width//16 + height//16 * width//16) * 8 * 8)
        init = torch.randn(shape, dtype=torch.float32, device='cuda')
        init_norm = torch.norm(init.view(batch_size, -1), 2.0, dim=1)
        normalized_init = init / init_norm[:, None]
        rand_norms = torch.pow(torch.rand(init.size()[0], dtype=torch.float32, device='cuda'), 1/shape[1])
        init = normalized_init * rand_norms[:, None]
        return init
        
    def _init(self, batch_size, height, width, eps):
        if self.rand_init:
            if self.opt == 'linf':
                cat_var = self._init_linf(batch_size, height, width)
            elif self.opt == 'l1':
                cat_var = self._init_l1(batch_size, height, width)
            elif self.opt == 'l2':
                cat_var = self._init_l2(batch_size, height, width)
            else:
                raise NotImplementedError
        else:
            cat_var = self._init_empty(batch_size, height, width)
        cat_var.mul_(eps[:, None])
        cat_var.requires_grad_()
        return cat_var

    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=False):
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
        batch_size, channels, height, width = pixel_img.size()
        if height % 16 != 0 or width % 16 != 0:
            raise Exception
        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True
        cat_var = self._init(batch_size, height, width, base_eps)
        if self.nb_its > 0:
            if self.opt in ['linf', 'l2']:
                cat_var = self._run_one_pgd(pixel_model, pixel_inp, cat_var, target,
                                            base_eps, step_size, avoid_target=avoid_target)
            elif self.opt == 'l1':
                cat_var = self._run_one_fw(pixel_model, pixel_inp, cat_var, target,
                                           base_eps, avoid_target=avoid_target)
            else:
                raise NotImplementedError
        # self.jpeg scales rounding_vars by base_eps, so we divide to rescale
        # its coordinates to [-1, 1]
        cat_var_temp = cat_var / base_eps[:, None]
        pixel_result = self._jpeg_cat(pixel_inp, cat_var_temp, base_eps, batch_size, height, width)
        return pixel_result

