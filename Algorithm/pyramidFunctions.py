import numpy as np
import cv2
from scipy.signal import convolve2d


def greduce(g0, a):
    k = np.transpose(np.array([[0.25 - 0.5 * a, 0.25, a, 0.25, 0.25 - 0.5 * a]]))
    h = np.matmul(k, k.T)
    P = np.zeros((g0.shape))
    P[:, :, 0] = convolve2d(g0[:, :, 0], h, boundary="fill", mode="same")
    P[:, :, 1] = convolve2d(g0[:, :, 1], h, boundary="fill", mode="same")
    P[:, :, 2] = convolve2d(g0[:, :, 2], h, boundary="fill", mode="same")
    g1 = P[::2, ::2, :].astype(np.float32)
    return g1


def gexpand(g1, a):
    k = np.transpose(np.array([[0.25 - 0.5 * a, 0.25, a, 0.25, 0.25 - 0.5 * a]]))
    h = np.matmul(k, k.T)
    n, m, o = g1.shape
    P = np.zeros((2 * n, 2 * m, 3))
    P[::2, ::2, 0] = g1[:, :, 0] * 4
    P[::2, ::2, 1] = g1[:, :, 1] * 4
    P[::2, ::2, 2] = g1[:, :, 2] * 4
    g0 = np.zeros((P.shape))
    g0[:, :, 0] = np.clip(convolve2d(P[:, :, 0], h, boundary="fill", mode="same"), 0, 1)
    g0[:, :, 1] = np.clip(convolve2d(P[:, :, 1], h, boundary="fill", mode="same"), 0, 1)
    g0[:, :, 2] = np.clip(convolve2d(P[:, :, 2], h, boundary="fill", mode="same"), 0, 1)
    g0 = g0.astype(np.float32)

    return g0


def im2gp(g0, nlevels):
    gp = list([g0])
    for i in range(0, nlevels):
        gp.append(greduce(gp[i], 0.4))
        # print(gp[i].shape)
    return gp


def im2lp(im, nlevels):
    gaussian = im2gp(im, nlevels)
    gaussian_len = len(gaussian)
    laplacian = [0] * (gaussian_len)
    laplacian[gaussian_len - 1] = gaussian[gaussian_len - 1]
    for i in range(gaussian_len - 1, -1, -1):
        if i == 0:
            break
        expanded_img = gexpand(gaussian[i], 0.4)
        if expanded_img.shape != gaussian[i - 1].shape:
            m, n, _ = gaussian[i - 1].shape
            expanded_img = cv2.resize(expanded_img, (n, m))
        laplacian[i - 1] = gaussian[i - 1] - expanded_img
    return laplacian


def lp2im(lp):
    lp_length = len(lp)
    recons = [0] * (lp_length - 1)  # Reconstruccion de imagen
    recons_len = len(recons)
    P = gexpand(lp[recons_len], 0.4)
    if P.shape != lp[recons_len - 1].shape:
        m, n, _ = lp[recons_len - 1].shape
        P = cv2.resize(P, (n, m))
    recons[recons_len - 1] = lp[recons_len - 1] + P

    for i in range(recons_len - 1, -1, -1):
        if i == 0:
            break
        expanded_img = gexpand(recons[i], 0.4)
        if expanded_img.shape != lp[i - 1].shape:
            m, n, _ = lp[i - 1].shape
            expanded_img = cv2.resize(expanded_img, (n, m))
        recons[i - 1] = lp[i - 1] + expanded_img

    im = np.copy(recons[0])
    return im


def imcompose(f1, f2, mask):
    # %Alpha compositing
    # %  g,      is the an output HxWx3 color image
    # g(x, y) = α(x, y)f1(x, y) + (1 − α(x, y)) f2(x, y)
    g = np.zeros((f1.shape))
    g[:, :, 0] = (mask * f1[:, :, 0]) + ((1 - mask) * f2[:, :, 0])
    g[:, :, 1] = (mask * f1[:, :, 1]) + ((1 - mask) * f2[:, :, 1])
    g[:, :, 2] = (mask * f1[:, :, 2]) + ((1 - mask) * f2[:, :, 2])

    g = g.astype(np.float32)
    return g


def imblend(im1, im2, mask, nlevels):
    n, m = mask.shape
    mask_mod = np.zeros((n, m, 3))
    mask_mod[:, :, 0] = mask
    mask_mod[:, :, 1] = mask
    mask_mod[:, :, 2] = mask
    f1 = im2lp(im1, nlevels)
    f2 = im2lp(im2, nlevels)
    fmask = im2gp(mask_mod, nlevels)

    mixed_img = [0] * len(f1)

    for i in range(0, len(f1)):
        mixed_img[i] = imcompose(np.double(f1[i]), np.double(f2[i]), fmask[i][:, :, 0])

    g = lp2im(mixed_img)
    return g
