import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

from subhkl.integration import Peaks
from subhkl.optimization import UB
from subhkl.config import beamlines

# def test_garnet():

a = b = c = 11.93
alpha = beta = gamma = 90
wavelength = [1, 3]
d_min = 0.7

cell = "Cubic"
centering = "I"

IPTS = 34720
instrument = "MANDI"
run_number = 12607
basename = "/SNS/{}/IPTS-{}/nexus/{}_{}.nxs.h5"
file = basename.format(instrument, IPTS, instrument, run_number)

directory = os.path.dirname(os.path.abspath(__file__))

peaks = Peaks(file, instrument)

file = os.path.basename(file)

name, ext = os.path.splitext(file)

# fname = os.path.join(directory, name + "_im.pdf")

# R, two_theta, az_phi, lamda = [], [], [], []

# with PdfPages(fname) as pdf:

#     for bank in sorted(peaks.ims.keys()):

#         i, j = peaks.harvest_peaks(bank, min_pix=15, min_rel_intens=0.2)
#         x, y = peaks.scale_coordinates(bank, i, j)

#         width, height = peaks.detector_width_height(bank)

#         im = peaks.ims[bank]

#         extent = [-width / 2, width / 2, -height / 2, height / 2]

#         tt, az = peaks.detector_trajectories(bank, i, j)

#         two_theta += tt.tolist()
#         az_phi += az.tolist()

#         R += [np.eye(3)] * len(tt)

#         lamda += [wavelength] * len(tt)

#         # ---

#         fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), layout="constrained")

#         ax.imshow(
#             im.T + 1,
#             norm="log",
#             cmap="binary",
#             origin="lower",
#             extent=extent,
#             # interpolation="none",
#             rasterized=True,
#         )

#         ax.set_aspect(1)
#         ax.minorticks_on()
#         ax.set_title(bank)

#         ax.set_xlabel("$x$ [m]")
#         ax.set_ylabel("$y$ [m]")

#         # ---

#         peak_dict = peaks.fit(i, j, im)

#         nx, ny = im.shape

#         for key in peak_dict.keys():

#             mu_1, mu_2, sigma_1, sigma_2, theta = peak_dict[key]

#             mu_x, mu_y = peaks.scale_coordinates(bank, mu_1, mu_2)
#             sigma_x, sigma_y, theta_xy = peaks.scale_ellipsoid(
#                 sigma_1, sigma_2, theta, width / nx, height / ny
#             )

#             ellipse = Ellipse(
#                 xy=(mu_x, mu_y),
#                 width=6 * sigma_x,
#                 height=6 * sigma_y,
#                 angle=np.rad2deg(theta_xy),
#                 linestyle="-",
#                 edgecolor="r",
#                 facecolor="none",
#                 rasterized=False,
#                 zorder=100,
#             )

#             ax.add_patch(ellipse)

#         pdf.savefig()
#         plt.close()

# # ---

ub = UB(a, b, c, alpha, beta, gamma)

#error, num, hkl, wl = ub.minimize(R, two_theta, az_phi, lamda)
#assert num / len(wl) > 0.5

ub.x = np.array([0.35463063, 0.34595509, 0.61926295])

fname = os.path.join(directory, name + "_com.pdf")

instrument_alt = "MANDIX"
basename = "/SNS/MANDI/IPTS-{}/shared/sipms/NOWX_{}.nxs.h5"

file_alt = basename.format(IPTS, run_number)

beamlines['MANDIX'] = {
    101: {
        "panel": "flat",
        'center': [-0.2451443483235737, 0.19161259685830377, -0.361994062121961534],
        'uhat': [0.7923095362185836, 0.5996606894734585, -0.11248402694298106],
        'vhat': [-0.35180984034106716, 0.5996607008892663, 0.7187745683092766],
        "width": 0.116,
        "height": 0.116,
        "m": 512,
        "n": 512,
        "offset": 4262144,
    },
    100: {
        "panel": "flat",
        'center': [0.2725144338398832, 0.165125968583036, -0.361994062194069637],
        'uhat': [0.3518098445351802, 0.5996606894734585, 0.7187745757804379],
        'vhat': [-0.7923095278170476, 0.5996607008892676, -0.11248402526278756],
        "width": 0.116,
        "height": 0.116,
        "m": 512,
        "n": 512,
        "offset": 4000000,
    },
}

peaks_alt = Peaks(file_alt, instrument_alt)

def downsample_sum(img, g=4):
    return img.reshape(im.shape[0] // g, g, im.shape[1] // g, g).sum(axis=(1, 3))

g = 1

fig, ax = plt.subplots(1, 2, figsize=(6.4*3, 6.4), gridspec_kw={'width_ratios': [3, 1]}, layout="constrained")

for ws in [peaks, peaks_alt]:

    h, k, l = ub.reflections(centering, d_min=d_min)
    xyz, hkl, wl, mult = ub.coverage(h, k, l, wavelength)
    h, k, l = hkl

    for bank in sorted(ws.ims.keys()):

        im = ws.ims[bank]

        if im.sum() > 0 and bank >= 10:

            vals = downsample_sum(im, g)

            i, j = np.meshgrid(np.arange(0, im.shape[0], g), np.arange(0, im.shape[1], g), indexing='ij')

            x, y, z = ws.transform_from_detector(bank, i.flatten(), j.flatten())

            l2 = np.sqrt(x**2 + y**2 + z**2)

            gamma = np.rad2deg(np.arctan2(x, z))
            nu = np.rad2deg(np.arcsin(y / l2))

            ax[0].scatter(gamma, nu, s=0.1, c=vals.flatten() + 1, vmin=1, vmax=100, cmap='binary', norm='log', rasterized=True)

            mask, i, j = ws.reflections_mask(bank, xyz)

            x, y, z = ws.transform_from_detector(bank, i, j)

            l2 = np.sqrt(x**2 + y**2 + z**2)

            gamma = np.rad2deg(np.arctan2(x, z))
            nu = np.rad2deg(np.arcsin(y / l2))

            counts = mult[mask]

            overlap = counts > 1

            hkl = np.array([h[mask], k[mask], l[mask]])

            ax[0].scatter(
                gamma[mask][~overlap], nu[mask][~overlap], s=20, edgecolor="r", facecolor="none", linewidth=0.2, zorder=10000000,
            )
            ax[0].scatter(
                gamma[mask][overlap], nu[mask][overlap], s=20, edgecolor="r", facecolor="none", linewidth=0.2, zorder=10000000,
            )

        elif im.sum() > 0:

            vals = downsample_sum(im, g)            

            i, j = np.meshgrid(np.arange(0, im.shape[0], g), np.arange(0, im.shape[1], g), indexing='ij')

            x, y, z = ws.transform_from_detector(bank, i.flatten(), j.flatten())

            ax[1].scatter(x * 100, z * 100, s=0.1, c=vals.flatten() + 1, vmin=1, vmax=100, cmap='binary', norm='log', rasterized=True)

            mask, i, j = ws.reflections_mask(bank, xyz)

            x, y, z = ws.transform_from_detector(bank, i, j)

            counts = mult[mask]

            overlap = counts > 1

            hkl = np.array([h[mask], k[mask], l[mask]])

            ax[1].scatter(
                x[mask][~overlap] * 100, z[mask][~overlap] * 100, s=20, edgecolor="r", facecolor="none", linewidth=0.2, zorder=10000000,
            )
            ax[1].scatter(
                x[mask][overlap] * 100, z[mask][overlap] * 100, s=20, edgecolor="r", facecolor="none", linewidth=0.2, zorder=10000000,
            )


ax[0].set_aspect(1)
ax[0].minorticks_on()
ax[0].set_title('$\mathrm{Yb_{3}Al_{5}O_{12}}$, MANDI')

ax[0].set_xlabel(r"$\gamma$")
ax[0].set_ylabel(r"$\nu$")

b100 = plt.Circle((-146, 34), 12, color='r', fill=False)
b101 = plt.Circle((+143, 30), 12, color='r', fill=False)

ax[0].add_patch(b100)
ax[0].add_patch(b101)
ax[0].invert_xaxis()

fmt_str_form = FormatStrFormatter(r"$%d^\circ$")

ax[0].xaxis.set_major_formatter(fmt_str_form)
ax[0].yaxis.set_major_formatter(fmt_str_form)

ax[1].set_aspect(1)
ax[1].minorticks_on()
#ax[1].set_title(r'')

ax[1].set_xlabel(r"$x$ [cm]")
ax[1].set_ylabel(r"$z$ [cm]")

fig.savefig(fname, bbox_inches='tight', dpi=300)

# test_garnet()
fname = os.path.join(directory, name + "_ind.pdf")

with PdfPages(fname) as pdf:

    for ws in [peaks, peaks_alt]:

        h, k, l = ub.reflections(centering, d_min=d_min)
        xyz, hkl, wl, mult = ub.coverage(h, k, l, wavelength)
        h, k, l = hkl

        for bank in sorted(ws.ims.keys()):

            mask, i, j = ws.reflections_mask(bank, xyz)
            x, y = ws.scale_coordinates(bank, i, j)

            width, height = ws.detector_width_height(bank)

            im = ws.ims[bank]

            if im.sum() > 0:

                extent = [-width / 2, width / 2, -height / 2, height / 2]
                counts = mult[mask]

                overlap = counts > 1

                hkl = np.array([h[mask], k[mask], l[mask]])
                xy = np.array([x[mask], y[mask]])

                fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), layout="constrained")

                ax.imshow(
                    im.T + 1,
                    norm="log",
                    cmap="binary",
                    origin="lower",
                    extent=extent,
                    # interpolation="none",
                    rasterized=True,
                )

                ax.scatter(
                    x[mask][~overlap] * 100, y[mask][~overlap] * 100, edgecolor="r", facecolor="r"
                )
                ax.scatter(
                    x[mask][overlap] * 100, y[mask][overlap] * 100, edgecolor="r", facecolor="none"
                )

                for i in range(hkl.shape[1]):
                    if np.linalg.norm(hkl[:, i]) > 0:
                        coord = xy[:, i]
                        label = "({:.0f},{:.0f},{:.0f})".format(*hkl[:, i])
                        ax.annotate(label, coord)

                ax.set_aspect(1)
                ax.minorticks_on()
                ax.set_title(bank)

                ax.set_xlabel("$x$ [cm]")
                ax.set_ylabel("$y$ [cm]")

                pdf.savefig()
                plt.close()

# # test_garnet()
