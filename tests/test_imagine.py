import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse

from subhkl.integration import Peaks
from subhkl.optimization import UB

# def test_garnet():

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

fname = os.path.join(directory, name + "_im.pdf")

R, two_theta, az_phi, lamda = [], [], [], []

for bank in sorted(peaks.ims.keys()):

    i, j = peaks.harvest_peaks(bank, min_pix=15, min_rel_intens=0.05)
    x, y = peaks.scale_coordinates(bank, i, j)

    width, height = peaks.detector_width_height(bank)

    im = peaks.ims[bank]

    extent = [-width / 2, width / 2, -height / 2, height / 2]

    tt, az = peaks.detector_trajectories(bank, i, j)

    two_theta += tt.tolist()
    az_phi += az.tolist()

    R += [np.eye(3)] * len(tt)

    lamda += [wavelength] * len(tt)

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
    ax.set_title(bank)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    # ---

    peak_dict = peaks.fit(i, j, im)

    nx, ny = im.shape

    for key in peak_dict.keys():

        mu_1, mu_2, sigma_1, sigma_2, theta = peak_dict[key]

        mu_x, mu_y = peaks.scale_coordinates(bank, mu_1, mu_2)
        sigma_x, sigma_y, theta_xy = peaks.scale_ellipsoid(
            sigma_1, sigma_2, theta, width / nx, height / ny
        )

        ellipse = Ellipse(
            xy=(mu_x, mu_y),
            width=6 * sigma_x,
            height=6 * sigma_y,
            angle=np.rad2deg(theta_xy),
            linestyle="-",
            edgecolor="r",
            facecolor="none",
            rasterized=False,
            zorder=100,
        )

        ax.add_patch(ellipse)

    fig.savefig(fname, bbox_inches='tight')

# ---

ub = UB(a, b, c, alpha, beta, gamma)

error, num, hkl, wl = ub.minimize(R, two_theta, az_phi, lamda)
assert num / len(wl) > 0.5

print(num / len(wl))

h, k, l = hkl.T
# h, k, l = ub.reflections(centering, d_min=d_min)
xyz, hkl, wl, mult = ub.coverage(h, k, l, wavelength)
h, k, l = hkl

fname = os.path.join(directory, name + "_ind.pdf")

for bank in sorted(peaks.ims.keys()):

    mask, i, j = peaks.reflections_mask(bank, xyz)
    x, y = peaks.scale_coordinates(bank, i, j)

    width, height = peaks.detector_width_height(bank)

    im = peaks.ims[bank]

    extent = [-width / 2, width / 2, -height / 2, height / 2]
    counts = mult[mask]

    overlap = counts > 1

    hkl = np.array([h[mask], k[mask], l[mask]])
    xy = np.array([x[mask], y[mask]])

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4), layout="constrained")
    
    ax.imshow(
        im.T,
        norm="log",
        cmap="binary",
        origin="lower",
        extent=extent,
        # interpolation="none",
        rasterized=True,
    )

    ax.scatter(
        x[mask][~overlap], y[mask][~overlap], edgecolor="r", facecolor="r"
    )
    ax.scatter(
        x[mask][overlap], y[mask][overlap], edgecolor="r", facecolor="none"
    )

    for i in range(hkl.shape[1]):
        if np.linalg.norm(hkl[:, i]) > 0:
            coord = xy[:, i]
            label = "({:.0f},{:.0f},{:.0f})".format(*hkl[:, i])
            ax.annotate(label, coord)

    ax.set_aspect(1)
    ax.minorticks_on()
    ax.set_title(bank)

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")

    fig.savefig(fname, bbox_inches="tight")

# test_garnet()
