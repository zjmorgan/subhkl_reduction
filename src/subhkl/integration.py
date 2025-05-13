import os
import re

import numpy as np

from h5py import File
from PIL import Image

import skimage.feature
import scipy.optimize

from subhkl.config import beamlines


class Peaks:

    def __init__(self, filename, instrument):
        """
        Find peaks from an image.

        Parameters
        ----------
        filename : str
            Filename of detector image.

        """

        name, ext = os.path.splitext(filename)

        self.instrument = instrument

        if ext == ".h5":
            self.ims = self.load_nexus(filename)
        else:
            self.ims = {0: np.array(Image.open(filename)).T}

    def load_nexus(self, filename):

        detectors = beamlines[self.instrument]

        ims = {}

        with File(filename, "r") as f:
            keys = f["/entry/"].keys()
            banks = [key for key in keys if re.search(r"bank\d", key)]

            for bank in banks:

                key = "/entry/" + bank + "/event_id"

                b = int(bank.split("bank")[1].split("_")[0])

                array = f[key][()]

                det = detectors.get(b)

                if det is not None:

                    m, n, offset = det["m"], det["n"], det["offset"]

                    bc = np.bincount(array - offset, minlength=m * n)

                    ims[b] = bc.reshape(m, n)

        return ims

    def harvest_peaks(self, bank, max_peaks=200, min_pix=50, min_rel_intens=0.5):
        """
        Locate peak positions in pixel coordinates.

        Parameters
        ----------
        bank : int
            Bank number.
        min_pix : int, optional
            Minimum pixel distance between peaks. The default is 50.
        min_rel_intens: float, optional
            Minimum intensity relative to maximum value. The default is 0.5

        Returns
        -------
        i : array, int
            x-pixel coordinates.
        j : array, int
            y-pixel coordinates.

        """

        coords = skimage.feature.peak_local_max(
            self.ims[bank],
            num_peaks=max_peaks,
            min_distance=min_pix,
            threshold_rel=min_rel_intens,
            exclude_border=min_pix*3,
        )

        return coords[:, 0], coords[:, 1]

    def scale_coordinates(self, bank, i, j):
        """
        Scale from pixel coordinates to real positions.

        Parameters
        ----------
        bank : int
            Bank number.
        i, j : array, int
            Image coordinates.
        scale_x, scale_y : float
            Pixel scaling factors.

        Returns
        -------
        x, y : array, float
            Image pixel position.

        """

        width, height = self.detector_width_height(bank)

        m, n = self.ims[bank].shape

        return (i / (m - 1) - 0.5) * width, (j / (n - 1) - 0.5) * height

    def scale_ellipsoid(self, a, b, theta, scale_x, scale_y):
        """
        Scale from pixel coordinates to real units.

        Parameters
        ----------
        a, b : array
            Image coordinates (eigenvalues).
        theta: array
            Orientation angle.
        scale_x, scale_y : float
            Pixel scaling factors.

        Returns
        -------
        x, y : array, float
            Image pixel size.

        """

        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        if np.isclose(a, 0) or np.isclose(b, 0):
            return 0, 0, 0

        S_inv = np.diag([1 / scale_x, 1 / scale_y])

        A = R.T @ np.diag([1 / a**2, 1 / b**2]) @ R

        A_new = S_inv.T @ A @ S_inv

        eigvals, eigvecs = np.linalg.eigh(A_new)

        new_a = 1 / np.sqrt(eigvals[0])
        new_b = 1 / np.sqrt(eigvals[1])

        new_theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        return new_a, new_b, new_theta

    def detector_width_height(self, bank):

        detector = beamlines[self.instrument][bank]

        width = detector["width"]
        height = detector["height"]

        return width, height

    def transform_from_detector(self, bank, i, j):

        detector = beamlines[self.instrument][bank]

        m = detector["m"]
        n = detector["n"]

        width = detector["width"]
        height = detector["height"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        u = np.array(i) / (m - 1) * width
        v = np.array(j) / (n - 1) * height

        dv = np.einsum("n,d->nd", v, vhat)

        if detector["panel"] == "flat":

            uhat = np.array(detector["uhat"])

            du = np.einsum("n,d->nd", u, uhat)

        else:

            radius = detector["radius"]
            rhat = np.array(detector["rhat"])

            w = np.cross(vhat, rhat)

            dvr = np.einsum("n,d->nd", radius * np.sin(u / radius), w)
            dr = np.einsum("n,d->nd", radius * (np.cos(u / radius) - 1), rhat)

            du = dr + dvr

        return (c + du + dv).T

    def transform_to_detector(self, bank, X, Y, Z):

        p = np.array([X, Y, Z])

        detector = beamlines[self.instrument][bank]

        m = detector["m"]
        n = detector["n"]

        width = detector["width"]
        height = detector["height"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        dw = width / (m - 1)
        dh = height / (n - 1)

        j = np.clip(np.dot(p.T - c, vhat) / dh, 0, n)

        if detector["panel"] == "flat":

            uhat = np.array(detector["uhat"])

            i = np.clip(np.dot(p.T - c, uhat) / dw, 0, m)

        else:

            radius = detector["radius"]
            rhat = np.array(detector["rhat"])

            d = p.T - c - (np.dot(p.T - c, vhat)[:,np.newaxis] * vhat)

            what = np.cross(vhat, rhat)

            dt = 2*np.arctan(-np.dot(d, rhat) / np.dot(d, what))
            dt = np.mod(dt, 2*np.pi)

            i = np.clip(dt * (radius / dw), 0, m)

        return i.astype(int), j.astype(int)

    def reflections_mask(self, bank, xyz):
        x, y, z = xyz

        detector = beamlines[self.instrument][bank]

        m = detector["m"]
        n = detector["n"]

        c = np.array(detector["center"])
        vhat = np.array(detector["vhat"])

        if detector["panel"] == "flat":

            uhat = np.array(detector["uhat"])
            norm = np.cross(uhat, vhat)

            d = np.einsum("i,in->n", norm, [x, y, z])
            t = np.dot(c, norm) / d

        else:

            radius = detector["radius"]

            d = np.einsum("i,in->n", vhat, [x, y, z])

            norm = np.sqrt((x-d*vhat[0])**2 + (y-d*vhat[1])**2 + (z-d*vhat[2])**2)

            t = radius/norm

        X, Y, Z = t * x, t * y, t * z

        i, j = self.transform_to_detector(bank, X, Y, Z)

        mask = (i > 0) & (j > 0) & (i < m - 1) & (j < n - 1) & (t > 0)

        return mask, i, j

    def detector_trajectories(self, bank, x, y):
        """
        Calculate detector trajectories.

        Parameters
        ----------
        bank : int
            Bank number.
        x, y : array, float
            Pixel position in physical units.

        Returns
        -------
        two_theta : array, float
            Scattering angles in degrees.
        az_phi : array, float
            Azimuthal angles in degrees.

        """

        X, Y, Z = self.transform_from_detector(bank, x, y)

        R = np.sqrt(X**2 + Y**2 + Z**2)
        two_theta = np.rad2deg(np.arccos(Z / R))
        az_phi = np.rad2deg(np.arctan2(Y, X))

        return two_theta, az_phi

    def peak(self, x, y, A, B, mu_x, mu_y, sigma_1, sigma_2, theta):

        a = 0.5 * (
            np.cos(theta) ** 2 / sigma_1**2 + np.sin(theta) ** 2 / sigma_2**2
        )
        b = 0.5 * (
            np.sin(theta) ** 2 / sigma_1**2 + np.cos(theta) ** 2 / sigma_2**2
        )
        c = 0.5 * (1 / sigma_1**2 - 1 / sigma_2**2) * np.sin(2 * theta)

        shape = np.exp(
            -(
                a * (x - mu_x) ** 2
                + b * (y - mu_y) ** 2
                + c * (x - mu_x) * (y - mu_y)
            )
        )

        return A * shape + B

    def residual(self, params, x, y, z):

        return (self.peak(x, y, *params) - z).flatten()

    def transform_ellipsoid(self, sigma_1, sigma_2, theta):

        sigma_x = np.hypot(sigma_1 * np.cos(theta), sigma_2 * np.sin(theta))
        sigma_y = np.hypot(sigma_1 * np.sin(theta), sigma_2 * np.cos(theta))
        rho = (
            (sigma_1**2 - sigma_2**2)
            * np.sin(2 * theta)
            / (2 * sigma_x * sigma_y)
        )

        return sigma_x, sigma_y, rho

    def intensity(self, A, B, sigma1, sigma2,  cov_matrix):

        I = A * 2 * np.pi * sigma1 * sigma2 - B
    
        dI = np.array([
            2 * np.pi * sigma1 * sigma2,
            -1,
            2 * np.pi * A * sigma2,
            2 * np.pi * A * sigma1,
        ])
    
        sigma = np.sqrt(dI @ cov_matrix @ dI.T)
    
        return I, sigma

    def fit(self, xp, yp, im, roi_pixels=50):

        peak_dict = {}

        X, Y = np.meshgrid(
            np.arange(im.shape[0]), np.arange(im.shape[1]), indexing="ij"
        )

        for ind, (x_val, y_val) in enumerate(zip(xp[:], yp[:])):

            x_min = int(max(x_val - roi_pixels, 0))
            x_max = int(min(x_val + roi_pixels + 1, im.shape[0]))

            y_min = int(max(y_val - roi_pixels, 0))
            y_max = int(min(y_val + roi_pixels + 1, im.shape[1]))

            x = X[x_min:x_max, y_min:y_max].copy()
            y = Y[x_min:x_max, y_min:y_max].copy()

            z = im[x_min:x_max, y_min:y_max].copy()

            x0 = (
                z.max(),
                z.min(),
                x_val,
                y_val,
                roi_pixels * 0.25,
                roi_pixels * 0.25,
                0,
            )

            xmin = (
                z.min(),
                0,
                x_val - roi_pixels * 0.5,
                y_val - roi_pixels * 0.5,
                1,
                1,
                -np.pi / 2,
            )

            xmax = (
                2 * z.max(),
                z.mean(),
                x_val + roi_pixels * 0.5,
                y_val + roi_pixels * 0.5,
                roi_pixels,
                roi_pixels,
                np.pi / 2,
            )

            bounds = np.array([xmin, xmax])

            args = (x, y, z)

            sol = scipy.optimize.least_squares(
                self.residual, x0=x0, bounds=bounds, args=args
            )

            J = sol.jac
            inv_cov = J.T.dot(J)

            if np.linalg.det(inv_cov) > 0:

                A, B, mu_1, mu_2, sigma_1, sigma_2, theta = sol.x

                inds = [0, 1, 4, 5]

                cov = np.linalg.inv(inv_cov)[inds][:,inds]

                I, sig = self.intensity(A, B, sigma_1, sigma_2, cov)

                if I < 3 * sig:
                    mu_1, mu_2 = x_val, y_val
                    sigma_1, sigma_2, theta = 0., 0., 0.

                items = mu_1, mu_2, sigma_1, sigma_2, theta

                peak_dict[(x_val, y_val)] = items

        return peak_dict
