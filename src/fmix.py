import math
import random
from typing import Union

import numpy as np
import torch
from scipy.stats import beta
from torch.nn import Module


def fftfreqnd(h: int, w: int = None, z: int = None) -> np.array:
    """ Get bin values for discrete fourier transform of size (h, w, z)

    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)


def fmix(images: torch.Tensor, targets: torch.Tensor, model: Module, criterion: Module) -> Module:
    decay_power = 3
    max_soft = 0
    batch_size, _, h, w = images.size()
    shape = (h, w)
    # generate mixed sample
    lam = np.random.beta(2.0, 1.0)
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)
    mask = torch.from_numpy(mask).float().cuda()
    rand_index = torch.randperm(batch_size)
    # When number of classes is small classes in shuffled batch is likely to coincide with original images
    # This is an attempt to acquire permutations with minimal number of coincidencies
    count = 0
    same = 100
    rand_index_final = []
    while torch.any(targets[rand_index] == targets):
        if count > 50:
            break
        rand_index = torch.randperm(batch_size)
        if same > torch.sum(targets[rand_index] == targets).item():
            same = torch.sum(targets[rand_index] == targets).item()
            rand_index_final = rand_index
        if same < 3:
            break
        count += 1
    # Mix the images
    images = mask * images + (1 - mask) * images[rand_index_final]
    shuffled_targets = targets[rand_index_final]

    outputs = model(images)
    return lam*criterion(outputs, targets) + (1-lam)*criterion(outputs, shuffled_targets), outputs


def get_spectrum(freqs: np.array, decay_power: float, ch, h: int, w: int = 0, z: int = 0) -> np.array:
    """ Samples a fourier image with given size and frequencies decayed by decay power

    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param


def make_low_freq_image(decay: float, shape: tuple, ch: int = 1) -> np.array:
    """ Sample a low frequency image from fourier space

    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask


def binarise_mask(mask: np.array, lam: float, in_shape: tuple, max_soft: float = 0.0) -> np.array:
    """ Binarises a given low frequency image such that it has mean lambda.

    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask
