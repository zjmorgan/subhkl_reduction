import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

from subhkl.integration import Peaks
from subhkl.optimization import UB

import scipy.ndimage
import skimage.feature

a = 18.39
b = 56.55
c = 6.54
alpha = beta = gamma = 90
wavelength = [2, 4.5]
d_min = 1.5

cell = "Orthorhombic"
centering = "F"

instrument = "IMAGINE"
folder = "/HFIR/CG4D/shared/images/ndip_data_test/meso_may/"
file = os.path.join(folder, "meso_2_15min_2-0_4-5_078.tif")

directory = os.path.dirname(os.path.abspath(__file__))

peaks = Peaks(file, instrument)

file = os.path.basename(file)

name, ext = os.path.splitext(file)

fname = os.path.join(directory, name + "_sphere.pdf")

R, two_theta, az_phi, lamda = [], [], [], []

for bank in sorted(peaks.ims.keys()):

    i, j = peaks.harvest_peaks(bank, min_pix=10, min_rel_intens=0.05)
    x, y = peaks.scale_coordinates(bank, i, j)

    width, height = peaks.detector_width_height(bank)

    im = peaks.ims[bank]

    extent = [-width / 2, width / 2, -height / 2, height / 2]

    tt, az = peaks.detector_trajectories(bank, i, j)

    mask = tt > 10#np.rad2deg(np.arcsin(0.5*wavelength[0]/np.max([a,b,c])))

    tt, az = tt[mask], az[mask]
    i, j = i[mask], j[mask]
    x, y = x[mask], y[mask]

    two_theta += tt.tolist()
    az_phi += az.tolist()

    R += [np.eye(3)] * len(tt)

    lamda += [wavelength] * len(tt)

    u = np.sin(np.deg2rad(tt))*np.cos(np.deg2rad(az))
    v = np.sin(np.deg2rad(tt))*np.sin(np.deg2rad(az))
    w = np.cos(np.deg2rad(tt))

    # ---

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), layout="constrained")

    ax.imshow(
        im.T,
        norm="log",
        cmap="binary",
        origin="lower",
        extent=extent,
        #interpolation="none",
        rasterized=True,
    )

    ax.set_aspect(1)
    ax.minorticks_on()
    # ax.set_title(bank)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    # peak_dict = peaks.fit(i, j, im, 25)

    # nx, ny = im.shape

    # for key in peak_dict.keys():

    #     mu_1, mu_2, sigma_1, sigma_2, theta = peak_dict[key]

    #     mu_x, mu_y = peaks.scale_coordinates(bank, mu_1, mu_2)
    #     sigma_x, sigma_y, theta_xy = peaks.scale_ellipsoid(
    #         sigma_1, sigma_2, theta, width / nx, height / ny
    #     )

    #     ellipse = Ellipse(
    #         xy=(mu_x, mu_y),
    #         width=6 * sigma_x,
    #         height=6 * sigma_y,
    #         angle=np.rad2deg(theta_xy),
    #         linestyle="-",
    #         edgecolor="r",
    #         facecolor="none",
    #         rasterized=True,
    #         zorder=100,
    #     )

    #     ax.add_patch(ellipse)

    mu_x, mu_y = peaks.scale_coordinates(bank, i, j)

    ax.scatter(x, y, s=5, color='r')

    fig.show()
    fig.savefig('imagine.mesolite.peak.finding.png', bbox_inches='tight')

    # ---

    X = u / (1 + np.sign(w) * w)
    Y = v / (1 + np.sign(w) * w)

    n_angles = 360
    angles = np.deg2rad(np.linspace(0, 180, n_angles, endpoint=False))

    n_bins = 400
    rho_max = 1
    rho_edges = np.linspace(-rho_max, rho_max, n_bins + 1)
    sinogram = np.zeros((n_bins, len(angles)))

    for i, theta in enumerate(angles):
        rho = X * np.cos(theta) + Y * np.sin(theta)
        counts, _ = np.histogram(rho, bins=rho_edges)
        sinogram[:, i] = counts

    coordinates = skimage.feature.peak_local_max(
        sinogram,
        min_distance=5,
        threshold_abs=sinogram.max() * 0.37
    )

    values = angles[coordinates[:, 1]]
    rho = rho_edges[:-1][coordinates[:, 0]]

    fig = plt.figure(figsize=(6.4, 6.4), layout='constrained')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.rad2deg(values), rho, c='r', s=1)

    ax.imshow(sinogram, extent=[0, 180, -rho_max, rho_max], aspect='auto', cmap='binary', origin='lower')
    ax.minorticks_on()
    
    formatter = FormatStrFormatter(r'$%d^\circ$')
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$r$')
    fig.show()
    fig.savefig('imagine.mesolite.radon.transform.png', bbox_inches='tight')

    fig = plt.figure(figsize=(6.4, 6.4), layout='constrained')
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X, Y, color='C0', rasterized=True)
    ax.minorticks_on()

    for r, theta in zip(rho, values):

        eps = 1e-8
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(sin_t) > eps:
            x_vals = np.linspace(-1, 1, 100)
            y_vals = (r - x_vals * cos_t) / sin_t
        else:
            y_vals = np.linspace(-1, 1, 100)
            x_vals = (r - y_vals * sin_t) / cos_t

        ax.plot(x_vals, y_vals, color='C1', zorder=0)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect(1)
    ax.set_xlabel(r'$x^\prime$')
    ax.set_ylabel(r'$y^\prime$')
    
    t = np.linspace(0, 2*np.pi, 500)
    ax.plot(np.cos(t), np.sin(t), color='k', linestyle='--', linewidth=1, zorder=-1)

    fig.show()
    fig.savefig('imagine.mesolite.stereographic.projection.png', bbox_inches='tight')

    fig = plt.figure(figsize=(6.4, 6.4), layout='constrained')

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(u, v, w, s=100, color='C0', rasterized=True)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)
    # ax.minorticks_on()

    for r, theta in zip(rho, values):

        eps = 1e-8
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(sin_t) > eps:
            x_vals = np.linspace(-100, 100, 5000)
            y_vals = (r - x_vals * cos_t) / sin_t
        else:
            y_vals = np.linspace(-100, 100, 5000)
            x_vals = (r - y_vals * sin_t) / cos_t

        U = 2 * x_vals / (1 + x_vals**2 + y_vals**2)
        V = 2 * y_vals / (1 + x_vals**2 + y_vals**2)
        W = (-1 + x_vals**2 + y_vals**2) / (1 + x_vals**2 + y_vals**2)

        ax.plot(U, V, W, color='C1', zorder=0)

        ax.plot(U, V, W, color='C1', zorder=0)

    ax.set_xlabel(r'$\hat{x}$')
    ax.set_ylabel(r'$\hat{y}$')
    ax.set_zlabel(r'$\hat{z}$')

    ax.plot(np.cos(t), np.sin(t), np.zeros_like(t), color='k', linestyle='--', linewidth=1, zorder=-1)
    # ax.plot(np.zeros_like(t), np.cos(t), np.sin(t), color='k', linestyle='--', linewidth=1, zorder=-1)
    # ax.plot(np.sin(t), np.zeros_like(t), np.cos(t), color='k', linestyle='--', linewidth=1, zorder=-1)

    fig.savefig('imagine.mesolite.peak.trajectories.png', bbox_inches='tight')

    fig.show()

    ip, jp = peaks.harvest_peaks(bank, min_pix=15, min_rel_intens=0.001)
    xp, yp = peaks.scale_coordinates(bank, ip, jp)

    width, height = peaks.detector_width_height(bank)

    im = peaks.ims[bank]

    extent = [-width / 2, width / 2, -height / 2, height / 2]

    ttp, azp = peaks.detector_trajectories(bank, ip, jp)

    maskp = ttp > 10# np.rad2deg(np.arcsin(0.5*wavelength[0]/np.max([a,b,c])))

    ttp, azp = ttp[maskp], azp[maskp]
    ip, jp = ip[maskp], jp[maskp]
    xp, yp = xp[maskp], yp[maskp]

    up = np.sin(np.deg2rad(ttp))*np.cos(np.deg2rad(azp))
    vp = np.sin(np.deg2rad(ttp))*np.sin(np.deg2rad(azp))
    wp = np.cos(np.deg2rad(ttp))

    # ---

    Xp = up / (1 + np.sign(wp) * wp)
    Yp = vp / (1 + np.sign(wp) * wp)

    line = Xp*np.cos(values[:, None]) + Yp*np.sin(values[:, None]) - rho[:, None]

    intersect = (np.abs(line) < 0.05).any(axis=0)

    ttp, azp = ttp[intersect], azp[intersect]
    ip, jp = ip[intersect], jp[intersect]
    xp, yp = xp[intersect], yp[intersect]

    # ---

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), layout="constrained")

    ax.imshow(
        im.T,
        norm="log",
        cmap="binary",
        origin="lower",
        extent=extent,
        #interpolation="none",
        rasterized=True,
    )

    ax.set_aspect(1)
    ax.minorticks_on()
    # ax.set_title(bank)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    mu_xp, mu_yp = peaks.scale_coordinates(bank, ip, jp)

    ax.scatter(xp, yp, s=5, color='w')

    mu_x, mu_y = peaks.scale_coordinates(bank, i, j)

    ax.scatter(x, y, s=5, color='r')

    fig.show()
    fig.savefig('imagine.mesolite.enhanced.peak.finding.png', bbox_inches='tight')
