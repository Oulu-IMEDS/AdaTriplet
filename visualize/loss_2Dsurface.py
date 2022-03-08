import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['text.usetex'] = True
n_rows = 14


def draw_loss(X, Y, Z, title, eps=0, beta=0, save=False, ignore_diag=False, draw_eps=True, draw_beta=False):
    global n_rows
    plt.set_cmap(cm.Spectral.reversed())
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['ytick.major.width'] = 0.5
    zz = Z[::-1, :]
    sz = zz.shape[0] - 1
    plt.figure(figsize=(1.8, 1.8), dpi=300)
    dx = cv.Sobel(zz, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(zz, cv.CV_64F, 0, 1, ksize=5)
    fig_img = plt.imshow(zz)

    m = zz.shape[0] // n_rows

    x = X[m // 2::m, m // 2::m]
    y = Y[m // 2::m, m // 2::m]
    xx = m * x.shape[1] * (x + 1) / 2
    yy = m * x.shape[0] * (y + 1) / 2
    u = dx[m // 2::m, m // 2::m]
    v = dy[m // 2::m, m // 2::m]

    if ignore_diag:
        u_flip = u[:, ::-1]
        v_flip = v[:, ::-1]
        np.fill_diagonal(u_flip, 0.0)
        np.fill_diagonal(v_flip, 0.0)
        u = u_flip[:, ::-1]
        v = v_flip[:, ::-1]

    plt.plot([sz, 0], [0, sz], color="w", linewidth=0.7, zorder=2)
    plt.plot([sz, eps * sz / 2], [eps * sz / 2, sz], color="w", linewidth=0.7, zorder=2)

    if draw_beta:
        plt.plot([0, sz], [(sz + 1) // 2, (sz + 1) // 2], color="w", linewidth=0.7, zorder=2)
        plt.plot([0, sz], [(sz + 1) // 2 - beta * sz / 2, (sz + 1) // 2 - beta * sz / 2], color="w", linewidth=0.7,
                 zorder=2)

    q = plt.quiver(xx, yy, -u, v, headwidth=7, pivot='middle', zorder=10, scale=30, scale_units='width')

    plt.tight_layout()
    plt.xticks([0, X.shape[0] // 2, X.shape[0] - 1], [-1, 0, 1])
    plt.yticks([0, Y.shape[0] // 2, Y.shape[0] - 1], [1, 0, -1])

    if draw_eps:
        plt.text(sz + 5, 23, r'$\varepsilon$')
    if draw_beta:
        plt.text(sz + 5, sz // 2 - beta, r'$\beta$')
    plt.ylabel('$\\phi_{an}$', labelpad=-1)
    plt.xlabel('$\\phi_{ap}$', labelpad=-1)
    plt.tight_layout()

    if save:
        plt.savefig(f"./figures/{title}.pdf")
    else:
        plt.title(title)
        plt.show()


def loss_trip_an(ap, an, eps, beta):
    return max(0, an - ap + eps) + max(0, an - beta)


def loss_trip(ap, an, eps):
    return max(0, an - ap + eps)


def draw_triplet_loss_surface(save=False):
    eps = 0.25

    l = 2
    gap = 2 / (2 * (100 // n_rows + 1) * n_rows)
    x = y = np.arange(-1.0, 1.0, gap)
    X, Y = np.meshgrid(x, y)
    zs_ours = np.array([loss_trip(x, y, eps) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z_ours = zs_ours.reshape(X.shape)
    title = 'Triplet'
    draw_loss(X, Y, Z_ours, title, eps=eps, draw_beta=False, save=save)


def draw_adatriplet_loss_surface(save=False):
    eps = 0.25
    beta = 0.1

    gap = 2 / (2 * (100 // n_rows + 1) * n_rows)
    x = y = np.arange(-1.0, 1.0, gap)
    X, Y = np.meshgrid(x, y)
    zs_ours = np.array([loss_trip_an(x, y, eps, beta) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z_ours = zs_ours.reshape(X.shape)
    title = 'AdaTriplet'
    draw_loss(X, Y, Z_ours, title, eps=eps, beta=beta, save=save,
              draw_eps=True, draw_beta=True)


if __name__ == "__main__":
    draw_triplet_loss_surface()
    draw_adatriplet_loss_surface()
