"""A galaxy catalog extracted from an HST image, as the input of a mock WCC field.

The CANDELS mosaics read by ``candels_cutout.py`` are images only: they carry no
source list. This module makes one from the cutout itself, so positions, sizes,
axis ratios and position angles are measured in the same pixel frame the image
is in, with no catalog angle conventions to convert:

    detect_hst_sources(data, wcs, zp_ab)  -> Table of sources (photutils)
    catalog_to_sersic(table, frame, ...)  -> SersicComponents + truth table

Each source becomes one single-Sersic component. Sersic indices are not
measured here: they default to n = 1 (an exponential disk) and are replaced by
the GIM2D fits of the HST/ACS Zurich (ZEST) catalog at IRSA where one exists
for a source (:func:`zest_sersic_n`; roughly a third of the galaxies brighter
than I = 24 have a finite GIM2D fit).

Not handled (prototype limits): one component per source however clumpy it is
(the clumps of a deblended galaxy come out as separate components when the
deblender splits it); the size correction for the HST PSF is a single
quadrature subtraction; the HST band (F606W) is taken to be the WCC band, as for
the Legacy Surveys mock.
"""

import os
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Table
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources

from legacysurvey_galaxy_mock import MAX_ELLIP, N_RANGE, pixel_position_angle
from wcc_sim import SersicComponent
from wcc_sim.starflux import REF_MAG, rate_for_ab, rate_for_spt

#: Half-light radius [arcsec] of the drizzled ACS/WFC F606W PSF, removed in
#: quadrature from the measured size (about FWHM / 2 for FWHM ~ 0.09").
HST_PSF_R_HALF_ARCSEC = 0.045

#: Smallest Sersic r_eff passed on [arcsec]: about two WCC pixels. Anything
#: smaller is effectively a point source and would only be sampled by its core.
MIN_R_EFF_ARCSEC = 0.035


def detect_hst_sources(data, wcs, zp_ab, pix_arcsec=0.030, nsigma=3.0, npixels=15,
                       min_snr=7.0, smooth_fwhm_px=3.0, deblend_contrast=0.05):
    """Sources in a sky-subtracted-ish HST image (e-/s), as an astropy Table.

    Detection is on a Gaussian-smoothed copy at `nsigma` times its noise, in at
    least `npixels` connected pixels, followed by deblending. Columns: x, y
    (pixel centroid), ra, dec, mag_ab (Kron), snr (Kron flux over its error),
    r_circ_arcsec (radius of the circle holding half the Kron flux), ellip
    (1 - b/a of the second moments), theta_pix (major axis, degrees CCW from
    +x of `data`), edge (the segment touches the image border).
    """
    data = np.asarray(data, dtype=float)
    bkg = Background2D(data, (64, 64), filter_size=(3, 3), bkg_estimator=MedianBackground())
    img = data - bkg.background
    _, _, sigma = sigma_clipped_stats(img, sigma=3)
    kernel = Gaussian2DKernel(smooth_fwhm_px * gaussian_fwhm_to_sigma, x_size=11, y_size=11)
    smooth = convolve(img, kernel, normalize_kernel=True)
    _, _, sigma_smooth = sigma_clipped_stats(smooth, sigma=3)
    segm = detect_sources(smooth, nsigma * sigma_smooth, npixels=npixels)
    if segm is None:
        return Table()
    segm = deblend_sources(smooth, segm, npixels=npixels, nlevels=32,
                           contrast=deblend_contrast, progress_bar=False)
    cat = SourceCatalog(img, segm, convolved_data=smooth, error=np.full_like(img, sigma))
    kron = np.asarray(cat.kron_flux, dtype=float)
    err = np.asarray(cat.kron_flux_err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = kron / err
        mag = zp_ab - 2.5 * np.log10(kron)
    a = np.asarray(cat.semimajor_sigma.value, dtype=float)
    b = np.asarray(cat.semiminor_sigma.value, dtype=float)
    x = np.asarray(cat.xcentroid, dtype=float)
    y = np.asarray(cat.ycentroid, dtype=float)
    ra, dec = wcs.pixel_to_world_values(x, y)
    ny, nx = img.shape
    bbox = np.array([[bb.ixmin, bb.ixmax, bb.iymin, bb.iymax] for bb in cat.bbox], dtype=float)
    edge = (bbox[:, 0] <= 0) | (bbox[:, 2] <= 0) | (bbox[:, 1] >= nx - 1) | (bbox[:, 3] >= ny - 1)
    out = Table({
        "x": x, "y": y, "ra": ra, "dec": dec, "mag_ab": mag, "snr": snr,
        "r_circ_arcsec": np.asarray(cat.fluxfrac_radius(0.5), dtype=float) * pix_arcsec,
        "ellip": 1.0 - b / a, "theta_pix": np.asarray(cat.orientation.to_value("deg"), dtype=float),
        "edge": edge,
    })
    good = np.isfinite(out["mag_ab"]) & np.isfinite(out["snr"]) & (out["snr"] >= min_snr) \
        & np.isfinite(out["ellip"]) & np.isfinite(out["r_circ_arcsec"]) & (out["r_circ_arcsec"] > 0)
    return out[good]


def mask_to_sources(data, nsigma=3.0, smooth_sigma_px=2.0, dilate_px=4, min_area_px=30):
    """`data` with everything away from detected light set to zero.

    ``ImageComponent`` clips negative pixels to zero, so the positive half of
    the HST pixel noise would otherwise enter the mock as sky signal, frozen
    in place (about as large as the noise of a 10 h WCC stack at these
    scales). Connected regions of the smoothed image above `nsigma` times its
    noise and at least `min_area_px` pixels large are kept, grown by
    `dilate_px`, noise inside them included; the rest becomes zero. The size
    cut matters: at 3 sigma about one island in 50 of that size is noise
    (estimated from the islands of the negative image), against about one in
    five at 2.5 sigma.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter, label

    data = np.asarray(data, dtype=float)
    smooth = gaussian_filter(np.nan_to_num(data), smooth_sigma_px)
    _, _, sigma = sigma_clipped_stats(smooth, sigma=3)
    labels, n = label(smooth > nsigma * sigma)
    if n == 0:
        return np.zeros_like(data)
    big = np.bincount(labels.ravel())[1:] >= min_area_px
    keep = binary_dilation(np.isin(labels, np.where(big)[0] + 1), iterations=dilate_px)
    return np.where(keep, data, 0.0)


def zest_sersic_n(ra, dec, radius_arcsec, cache_dir=None, match_arcsec=0.4):
    """Sersic index from the ACS Zurich (ZEST) GIM2D fits at IRSA, NaN where none.

    `ra`, `dec` are arrays; the query is one box around them. Needs network on
    the first call (cached as ECSV).
    """
    ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
    n = np.full(len(ra), np.nan)
    if len(ra) == 0:
        return n
    centre = SkyCoord(ra.mean() * u.deg, dec.mean() * u.deg)
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"zest_{centre.ra.deg:.5f}_{centre.dec.deg:+.5f}_{radius_arcsec:.0f}.ecsv")
    if path is not None and os.path.exists(path):
        zest = Table.read(path, format="ascii.ecsv")
    else:
        from astroquery.ipac.irsa import Irsa

        Irsa.TIMEOUT = 120
        zest = Irsa.query_region(centre, catalog="cosmos_morph_zurich_1", spatial="Box",
                                 width=2.0 * radius_arcsec * u.arcsec)
        zest = zest["ra", "dec", "sersic_n_gim2d", "r_gim2d", "acs_mag_auto"]
        if path is not None:
            zest.write(path, format="ascii.ecsv", overwrite=True)
    if len(zest) == 0:
        return n
    zn = np.ma.filled(np.ma.asarray(zest["sersic_n_gim2d"], dtype=float), np.nan)
    idx, sep, _ = match_coordinates_sky(
        SkyCoord(ra * u.deg, dec * u.deg), SkyCoord(zest["ra"], zest["dec"], unit="deg"))
    ok = sep.arcsec < match_arcsec
    n[ok] = zn[idx[ok]]
    return n


def catalog_to_sersic(table, wcs, frame, sensorfilter, n_sersic=None, template="G2V",
                      drop_edge=True, gaia=None, gaia_match_arcsec=0.5):
    """One SersicComponent per HST source; also returns the truth table.

    `wcs` is the WCS of the image the table was measured on and `frame` the
    detector frame of the simulation. `n_sersic` (array, NaN = unknown) overrides
    the default exponential n = 1. Sources touching the image border (cut off) are
    dropped when `drop_edge`; sources within `gaia_match_arcsec` of a row of the
    `gaia` table are dropped because the simulator already draws that star.
    """
    t = table
    keep = np.ones(len(t), dtype=bool)
    if drop_edge:
        keep &= ~np.asarray(t["edge"], dtype=bool)
    if gaia is not None and len(gaia):
        _, sep, _ = match_coordinates_sky(
            SkyCoord(t["ra"] * u.deg, t["dec"] * u.deg),
            SkyCoord(np.asarray(gaia["ra"], dtype=float) * u.deg,
                     np.asarray(gaia["dec"], dtype=float) * u.deg))
        keep &= sep.arcsec > gaia_match_arcsec
    t = t[keep]
    n = np.ones(len(t)) if n_sersic is None else np.where(np.isfinite(np.asarray(n_sersic)[keep]),
                                                          np.asarray(n_sersic)[keep], 1.0)
    n = np.clip(n, *N_RANGE)

    ellip = np.clip(np.asarray(t["ellip"], dtype=float), 0.0, MAX_ELLIP)
    q = 1.0 - ellip
    # r_circ holds half the light inside a circle; for an ellipse the semi-major
    # half-light radius is r_circ / sqrt(q). Then remove the HST PSF.
    r_major = np.asarray(t["r_circ_arcsec"], dtype=float) / np.sqrt(q)
    r_eff = np.sqrt(np.clip(r_major**2 - HST_PSF_R_HALF_ARCSEC**2, 0.0, None))
    r_eff = np.clip(r_eff, MIN_R_EFF_ARCSEC, None)

    # position angle: HST pixel angle -> sky (E of N) through the HST WCS -> WCC pixel angle
    x, y = np.asarray(t["x"], dtype=float), np.asarray(t["y"], dtype=float)
    th = np.radians(np.asarray(t["theta_pix"], dtype=float))
    p0 = SkyCoord(*wcs.pixel_to_world_values(x, y), unit="deg")
    p1 = SkyCoord(*wcs.pixel_to_world_values(x + np.cos(th), y + np.sin(th)), unit="deg")
    pa_sky = p0.position_angle(p1).deg % 180.0
    ra, dec = np.asarray(t["ra"], dtype=float), np.asarray(t["dec"], dtype=float)
    pa_pix = pixel_position_angle(frame, ra, dec, pa_sky)

    m_ab = np.asarray(t["mag_ab"], dtype=float)
    rate_spt = rate_for_spt(template, sensorfilter)
    g_equiv = REF_MAG - 2.5 * np.log10(
        np.array([rate_for_ab(m, sensorfilter) for m in m_ab]) / rate_spt)

    comps = [
        SersicComponent(ra=float(ra[i]), dec=float(dec[i]), n=float(n[i]),
                        r_eff_arcsec=float(r_eff[i]), ellip=float(ellip[i]),
                        pa_deg=float(pa_pix[i]), total_mag=float(g_equiv[i]), template=template)
        for i in range(len(t))
    ]
    truth = Table({
        "ra": ra, "dec": dec, "n": n, "r_eff_arcsec": r_eff, "ellip": ellip,
        "pa_sky_deg": pa_sky, "pa_pix_deg": pa_pix, "mag_ab_band": m_ab, "snr": np.asarray(t["snr"]),
        "total_mag_G": g_equiv,
    })
    return comps, truth


def count_summary(table, comps):
    """One line describing what was extracted and kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mags = np.asarray(table["mag_ab"], dtype=float)
    return (f"{len(table)} HST sources (S/N cut), {len(comps)} kept as Sersic components; "
            f"AB mag 10/50/90%: {np.nanpercentile(mags, [10, 50, 90]).round(1).tolist()}")
