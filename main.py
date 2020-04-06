import skimage as sk
import skimage.io as io
import numpy as np
import scipy.signal as signal
import scipy.ndimage.interpolation as interpolation
import cv2
import matplotlib.pyplot as plt
import math
import skimage.transform as sktr
import sys
from skimage.util import crop

edge_fil = np.array([[1, -1]])


def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im, [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, multichannel=True)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, multichannel=True)
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    print(im1.shape)
    h2, w2, c2 = im2.shape
    print(im2.shape)
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): -int(np.ceil((h2 - h1) / 2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): -int(np.ceil((h1 - h2) / 2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): -int(np.ceil((w2 - w1) / 2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)): -int(np.ceil((w1 - w2) / 2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    print(im1.shape, im2.shape, 'after center')
    im1, im2 = rescale_images(im1, im2, pts)
    print(im1.shape, im2.shape, 'after rescale')
    im1, angle = rotate_im1(im1, im2, pts)
    print(im1.shape, im2.shape, 'after rotate')
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def binarize(img, th):
    return img > th


def getD_x(img):
    d_x = signal.convolve2d(img, edge_fil, 'same')
    return d_x


def getD_y(img):
    d_y = signal.convolve2d(img, np.transpose(edge_fil), 'same')
    return d_y


def get_mag_edge(img, threshold):
    D_x = getD_x(img)
    D_y = getD_y(img)
    result = np.sqrt(D_x ** 2 + D_y ** 2)
    result = result - np.min(result)
    result = result / (np.max(result) - np.min(result))
    result = binarize(result, threshold)
    return result


def smooth(img):
    return signal.convolve2d(img, gau(), 'same')


def normalize(img):
    new_img = img - np.min(img)
    new_img = new_img / (np.max(new_img) - np.min(new_img))
    return new_img


def gau():
    gau = cv2.getGaussianKernel(9, -1)
    gau = np.outer(gau, gau.T)
    return gau


def rotate(img, angle):
    return interpolation.rotate(img, angle)


def center(img):
    h, w = img.shape
    h /= 2
    h = int(h)
    w /= 2
    w = int(w)
    return img[h - 300:h + 300, w - 300:w + 300]


def comp_angle(img):
    b, g, r = cv2.split(img)
    size = center(b).shape
    dx, dy = np.zeros(size), np.zeros(size)
    for chan in [b, g, r]:
        chan = center(chan)
        chan = smooth(chan)
        d_x = getD_x(chan)
        d_y = getD_y(chan)
        dx += d_x
        dy += d_y
    dx /= 3
    dx += 0.01
    dy /= 3
    angle_m = dy / dx
    tan = np.arctan(angle_m)
    return tan


def count_straight(img):
    angles = comp_angle(img)
    summ = 0
    summ += np.sum(np.abs(angles - 0) < 0.005)
    summ += np.sum(angles > 1.56)
    summ += np.sum(angles < -1.56)
    return summ


def find_angle(img):
    best_deg, best_score = 0, count_straight(img)
    for deg in np.arange(-10, 10, 0.5):
        if deg != 0:
            rot = rotate(img, deg)
            score = count_straight(rot)
            if score > best_score:
                best_score = score
                best_deg = deg
    return best_deg


def sharp1(img):
    img = normalize(img)
    unsharp = sk.filters.unsharp_mask
    b, g, r = cv2.split(img)
    r = unsharp(r)
    b = unsharp(b)
    g = unsharp(g)
    re = np.stack([b, g, r], axis=2)
    return re


def gau_pyramid(img, layer, track, sigma):
    if layer == 0:
        return track
    else:
        fil = cv2.getGaussianKernel(3 * sigma, sigma)
        fil = np.outer(fil, fil.T)
        if len(img.shape) == 3 and img.shape[2] == 3:
            b, g, r = cv2.split(img)
            r = signal.convolve2d(r, fil, 'same')
            g = signal.convolve2d(g, fil, 'same')
            b = signal.convolve2d(b, fil, 'same')
            after = np.dstack([b, g, r])
            track.append(after)
            return gau_pyramid(after, layer - 1, track, sigma)
        elif len(img.shape) == 2:
            after = signal.convolve2d(img, fil, 'same')
            track.append(after)
            return gau_pyramid(after, layer - 1, track, sigma)


def lap_pyramid(img, layer, sigma):
    gau = gau_pyramid(img, layer, [], sigma)
    track = []
    track.append(img - gau[0])
    for i in range(layer - 1):
        track.append(gau[i] - gau[i + 1])
    return track


def blend(im1, im2, horizontal=True, mask=None):
    assert im1.shape == im2.shape
    h, w, c = im1.shape
    lap_p1 = lap_pyramid(im1, 5, 2)
    lap_p2 = lap_pyramid(im2, 5, 2)
    gau_p1 = gau_pyramid(im1, 5, [], 2)
    gau_p2 = gau_pyramid(im2, 5, [], 2)
    lap_p1.append(gau_p1[-1])
    lap_p2.append(gau_p2[-1])
    gau = cv2.getGaussianKernel(60, 30)
    gau = np.outer(gau, gau.T)
    if mask is None:
        mask = np.zeros(im1.shape[:2])
        if horizontal:
            for i in range(h):
                for j in range(w):
                    if j < w / 2:
                        mask[i, j] = 1
        else:
            for i in range(h):
                for j in range(w):
                    if i > h / 2:
                        mask[i, j] = 1

    result = np.zeros(im1.shape)
    for i in range(len(lap_p1)):
        result += level_blend(lap_p1[i], lap_p2[i], mask)
        mask = signal.convolve2d(mask, gau, 'same')
    return result


def level_blend(im1, im2, mask):
    return apply_mask(im2, mask) + apply_mask(im1, 1 - mask)


def apply_mask(img, mask):
    b, g, r = cv2.split(img)
    b = b * mask
    g = g * mask
    r = r * mask
    return np.dstack([b, g, r])


def hybrid(imname1, imname2):
    im1 = plt.imread(imname1) / 255.
    im2 = plt.imread(imname2) / 255
    im1_aligned, im2_aligned = align_images(im1, im2)

    sigma1 = 3
    sigma2 = 4
    gauH = cv2.getGaussianKernel(10, sigma1)
    gauL = cv2.getGaussianKernel(15, sigma2)
    gauH = np.outer(gauH, gauH.T)
    gauL = np.outer(gauL, gauL.T)

    def getLowF(img, fil):
        b, g, r = cv2.split(img)
        store = np.zeros(b.shape)
        for chan in [b, g, r]:
            chan = signal.convolve2d(chan, fil, 'same')
            store += chan
        store /= 3
        return store

    def getHighF(img, fil):
        b, g, r = cv2.split(img)
        store = np.zeros(b.shape)
        for chan in [b, g, r]:
            lowF = signal.convolve2d(chan, fil, 'same')
            chan = chan - lowF
            store += chan
        store /= 3
        return store

    lowF = getLowF(im2_aligned, gauL)
    highF = getHighF(im1_aligned, gauH)
    combined = 0.5 * lowF + 1 * highF
    return combined


if __name__ == "__main__":
    if len(sys.argv) == 3 and (sys.argv[2] == 'x' or sys.argv[2] == 'y' or sys.argv[2] == 'e'):
        impath = sys.argv[1]
        img = io.imread(impath)
        if len(img.shape) != 2:
            img = cv2.split(img)[0]
            if sys.argv[2] == 'x':
                result = getD_x(img)
                result = normalize(result)
                io.imsave('D_x.jpg', result)
            elif sys.argv[2] == 'y':
                result = getD_y(img)
                result = normalize(result)
                io.imsave('D_y.jpg', result)
            if sys.argv[2] == 'e':
                result = get_mag_edge(img, 0.085).astype(np.uint8)
                result = normalize(result)
                io.imsave('Edge_mag.jpg', result)
        else:
            if sys.argv[2] == 'x':
                result = getD_x(img)
                io.imsave('D_x.jpg', result)
            elif sys.argv[2] == 'y':
                result = getD_y(img)
                io.imsave('D_y.jpg', result)
            if sys.argv[2] == 'e':
                result = get_mag_edge(img, 0.085)
                io.imsave('Edge_mag.jpg', result)
    elif len(sys.argv) == 3 and (sys.argv[2] == 'gaux' or sys.argv[2] == 'gauy' or sys.argv[2] == 'gaue'):
        impath = sys.argv[1]
        img = io.imread(impath)
        if len(img.shape) != 2:
            img = cv2.split(img)[0]
            img = smooth(img)
            if sys.argv[2] == 'gaux':
                result = getD_x(img)
                result = normalize(result)
                io.imsave('D_x Gau.jpg', result)
            elif sys.argv[2] == 'gauy':
                result = getD_y(img)
                result = normalize(result)
                io.imsave('D_y Gau.jpg', result)
            if sys.argv[2] == 'gaue':
                result = get_mag_edge(img, 0.085).astype(np.uint8)
                result = normalize(result)
                io.imsave('Edge_mag Gau.jpg', result)
        else:
            img = smooth(img)
            if sys.argv[2] == 'gaux':
                result = getD_x(img)
                result = normalize(result)
                io.imsave('D_x Gau.jpg', result)
            elif sys.argv[2] == 'gauy':
                result = getD_y(img)
                result = normalize(result)
                io.imsave('D_y Gau.jpg', result)
            if sys.argv[2] == 'gaue':
                result = get_mag_edge(img, 0.085).astype(np.uint8)
                result = normalize(result)
                io.imsave('Edge_mag Gau.jpg', result)


    elif len(sys.argv) == 3 and (sys.argv[2] == 'straight'):
        impath = sys.argv[1]
        img = io.imread(impath)
        deg = find_angle(img)
        print("Angle computed:", deg)
        io.imsave('straight_'+impath, rotate(img, deg))

    elif len(sys.argv) == 3 and (sys.argv[2] == 'sharp'):
        impath = sys.argv[1]
        img = io.imread(impath)
        result = normalize(sharp1(img))
        io.imsave('sharp_' + impath, result)

    elif len(sys.argv) == 4 and (sys.argv[3] == 'hybrid'):
        impath1 = sys.argv[1]
        impath2 = sys.argv[2]
        result = hybrid(impath1, impath2)
        print(impath1[:3] + impath2[:3] + "_hybrid.jpg")
        io.imsave(impath1[:3] + impath2[:3] + "_hybrid.jpg", result)

    elif len(sys.argv) == 3 and (sys.argv[2] == "gaupyramid" or sys.argv[2] == "lappyramid"):
        impath = sys.argv[1]
        img = io.imread(impath)
        if sys.argv[2] == "gaupyramid":
            gau_p = gau_pyramid(img, 5, [], 15)
            for i in range(5):
                io.imsave(str(i+1)+"gau"+impath, gau_p[i])
        elif sys.argv[2] == "lappyramid":
            lap_p = lap_pyramid(img, 5, 15)
            for i in range(5):
                io.imsave(str(i+1)+"lap"+impath, lap_p[i])

    elif len(sys.argv)==5 and sys.argv[3] == "blend" and (sys.argv[4] == 'h' or sys.argv[4] == 'v'):
        impath1 = sys.argv[1]
        impath2 = sys.argv[2]
        img1 = io.imread(impath1)
        img2 = io.imread(impath2)
        result = blend(img1, img2, sys.argv[4] == "h", None)
        io.imsave(impath1[:3] + impath2[:3] + "_blend.jpg", result)


