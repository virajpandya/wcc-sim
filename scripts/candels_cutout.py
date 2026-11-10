"""Small cutouts from the CANDELS-COSMOS HST mosaics, without downloading them.

The mosaics (``cos-tot``, v1.0, MAST HLSP) are 0.8-8 GB each, but a FITS image
is one contiguous block of rows, so the rows a cutout needs can be read with a
single HTTP byte-range request and sliced in memory. The result is cached as a
small FITS file with a WCS.

    from candels_cutout import fetch_candels_cutout
    data, wcs, zp_ab, header = fetch_candels_cutout(150.12, 2.20, 52.0, 35.0, "f606w",
                                                    cache_dir="ls_cache")

Bands in COSMOS: ACS ``f606w`` / ``f814w`` (30 mas/px) and WFC3 ``f125w`` /
``f160w`` (60 mas/px). The footprint is about 14' x 36' around
(150.12, +2.35); a position outside it raises ValueError.

Data are in e-/s; ``zp_ab`` converts a pixel value v to AB magnitude as
``zp_ab - 2.5 log10(v)`` (see :func:`wcc_sim.phangs.ab_zeropoint`).
"""

import os
import warnings

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

BASE = "https://archive.stsci.edu/pub/hlsp/candels/cosmos/cos-tot/v1.0/"
FILES = {
    "f606w": "hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits",
    "f814w": "hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits",
    "f125w": "hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits",
    "f160w": "hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits",
}
_BLOCK = 2880
_TIMEOUT = 600


def _read_primary_header(url):
    """(header, byte offset of the data) from the first few KB of the file."""
    size = 20 * _BLOCK
    while True:
        r = requests.get(url, headers={"Range": f"bytes=0-{size - 1}"}, timeout=_TIMEOUT)
        r.raise_for_status()
        raw = r.content
        for k in range(0, len(raw), _BLOCK):
            if b"END" + b" " * 77 in [raw[k + i:k + i + 80] for i in range(0, _BLOCK, 80)]:
                end = k + _BLOCK
                return fits.Header.fromstring(raw[:end].decode("ascii")), end
        if len(raw) < size:
            raise ValueError(f"no END card found in {url}")
        size *= 4


def _celestial_wcs(header):
    """WCS without the SIP terms.

    The drizzled mosaics already have distortion removed (CTYPE is a plain
    TAN) but still carry the SIP keywords of the input frames, which astropy
    would apply and shift positions by.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(header, naxis=2)
    wcs.sip = None
    return wcs


def fetch_candels_cutout(ra, dec, width_arcsec, height_arcsec, band="f606w", cache_dir=None):
    """Cutout of the CANDELS-COSMOS mosaic centred on (ra, dec).

    Returns ``(data, wcs, zp_ab, header)``: `data` is (ny, nx) in e-/s with the
    pixel grid of the mosaic (north up), `wcs` its celestial WCS.
    """
    band = band.lower()
    if band not in FILES:
        raise ValueError(f"band must be one of {sorted(FILES)}, got {band!r}")
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir, f"candels_cos_{band}_{ra:.5f}_{dec:+.5f}_{width_arcsec:.1f}x{height_arcsec:.1f}.fits")
    if path is not None and os.path.exists(path):
        with fits.open(path) as hdul:
            data, header = np.array(hdul[0].data, dtype=float), hdul[0].header.copy()
        return data, _celestial_wcs(header), float(header["ZPAB"]), header

    url = BASE + FILES[band]
    header, offset = _read_primary_header(url)
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    if header["BITPIX"] != -32:
        raise ValueError(f"unexpected BITPIX {header['BITPIX']} in {FILES[band]}")
    wcs = _celestial_wcs(header)
    scale = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0
    xc, yc = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
    hx, hy = 0.5 * width_arcsec / scale, 0.5 * height_arcsec / scale
    x0, x1 = int(round(xc - hx)), int(round(xc + hx))
    y0, y1 = int(round(yc - hy)), int(round(yc + hy))
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        raise ValueError(
            f"({ra}, {dec}) with a {width_arcsec}\" x {height_arcsec}\" patch is outside the "
            f"CANDELS-COSMOS {band} mosaic (x {x0}..{x1} of {nx}, y {y0}..{y1} of {ny})")

    row_bytes = nx * 4
    first, last = offset + y0 * row_bytes, offset + y1 * row_bytes - 1
    r = requests.get(url, headers={"Range": f"bytes={first}-{last}"}, timeout=_TIMEOUT)
    r.raise_for_status()
    if r.status_code != 206 or len(r.content) != last - first + 1:
        raise RuntimeError(f"byte-range read failed (status {r.status_code}, {len(r.content)} bytes)")
    band_rows = np.frombuffer(r.content, dtype=">f4").reshape(y1 - y0, nx)
    data = band_rows[:, x0:x1].astype(np.float32)

    zp_ab = -2.5 * np.log10(header["PHOTFLAM"]) - 5.0 * np.log10(header["PHOTPLAM"]) - 2.408
    out = _celestial_wcs(header)
    out.wcs.crpix = out.wcs.crpix - [x0, y0]
    hdr = out.to_header()
    hdr["BAND"] = band.upper()
    hdr["BUNIT"] = "ELECTRONS/S"
    hdr["ZPAB"] = (float(zp_ab), "AB zero point: m = ZPAB - 2.5 log10(e-/s)")
    hdr["SOURCE"] = FILES[band]
    if path is not None:
        fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)
    return data.astype(float), _celestial_wcs(hdr), float(zp_ab), hdr
