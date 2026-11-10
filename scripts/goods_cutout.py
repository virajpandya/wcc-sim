"""Small cutouts from the original GOODS ACS mosaics (F435W, F606W, F775W, F850LP).

The CANDELS mosaics have no blue ACS band. The GOODS v2.0 release (Giavalisco et al. 2004; MAST HLSP ``goods/v2``) has ACS **F435W (B), F606W (V),
F775W (i) and F850LP (z)** over both GOODS fields at 30 mas/pixel, and the Hubble Deep Field North and the Ultra Deep Field lie inside them. B, V and i
are the closest HST match to the Lazuli WCC g, r and i filters.

    from goods_cutout import fetch_goods_cutout
    data, wcs, zp_ab, header = fetch_goods_cutout(53.16, -27.82, 180.0, 180.0, "f435w", field="goods-s", cache_dir="ls_cache")

The mosaics are 8192 x 8192 pixel tiles ("sections") on one common 30 mas grid. A cutout is read from the one tile that contains it, with the byte-range
method of ``candels_cutout.cutout_from_url``; a patch that crosses a tile boundary raises ValueError (move its centre or make it smaller). The tiles have no
exposure map: ``TYPICAL_EXPOSURE_S`` gives the net exposure of a typical region, from the GOODS v2.0 documentation.

Pixel values are in e-/s; ``zp_ab`` converts a pixel value v to AB magnitude as ``zp_ab - 2.5 log10(v)`` (the zero points of the GOODS v2.0 documentation; the
PHOTFLAM keywords of the headers are zero).
"""

import functools
import os
import re

import numpy as np
from astropy.io import fits

import candels_cutout as cc

BASE = "https://archive.stsci.edu/pub/hlsp/goods/v2/"
FIELDS = {"goods-n": "n", "goods-s": "s"}
BANDS = {"f435w": "b", "f606w": "v", "f775w": "i", "f850lp": "z"}

#: AB zero points of the v2.0 science images (e-/s), from the GOODS v2.0 documentation.
ZP_AB = {"f435w": 25.673, "f606w": 26.486, "f775w": 25.654, "f850lp": 24.862}

#: Net exposure [s] in a typical region of the v2.0 mosaics (GOODS v2.0 documentation).
TYPICAL_EXPOSURE_S = {
    "goods-n": {"f435w": 7200.0, "f606w": 5650.0, "f775w": 8530.0, "f850lp": 24760.0},
    "goods-s": {"f435w": 7200.0, "f606w": 5450.0, "f775w": 7028.0, "f850lp": 18232.0},
}
_KINDS = {"drz": "drz", "wht": "wht"}


def _check(field, band, kind="drz"):
    field, band = str(field).lower(), str(band).lower()
    if field not in FIELDS:
        raise ValueError(f"field must be one of {sorted(FIELDS)}, got {field!r}")
    if band not in BANDS:
        raise ValueError(f"band must be one of {sorted(BANDS)}, got {band!r}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    return field, band


def tile_url(field, band, section, kind="drz"):
    """URL of one tile, e.g. ``section="sect23"``."""
    field, band = _check(field, band, kind)
    return f"{BASE}h_{FIELDS[field]}{BANDS[band]}_{section}_v2.0_{_KINDS[kind]}_img.fits"


@functools.lru_cache(maxsize=None)
def list_sections(field, band):
    """The section names of a field and band (``"sect11"``, ...), from the archive listing."""
    field, band = _check(field, band)
    page = cc._get(BASE).text
    pattern = rf'href="h_{FIELDS[field]}{BANDS[band]}_(sect\d+)_v2\.0_drz_img\.fits"'
    return tuple(sorted(set(re.findall(pattern, page))))


@functools.lru_cache(maxsize=None)
def _tile_geometry(field, band, section):
    header, _ = cc._read_primary_header(tile_url(field, band, section))
    return int(header["NAXIS1"]), int(header["NAXIS2"]), cc._celestial_wcs(header)


def find_tile(field, band, ra, dec, width_arcsec, height_arcsec):
    """Section whose tile contains the whole patch centred on (ra, dec)."""
    field, band = _check(field, band)
    partial = []
    for section in list_sections(field, band):
        nx, ny, wcs = _tile_geometry(field, band, section)
        scale = cc._pixel_scale_arcsec(wcs)
        x, y = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        hx, hy = 0.5 * width_arcsec / scale, 0.5 * height_arcsec / scale
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if hx <= x <= nx - hx and hy <= y <= ny - hy:
            return section
        if -hx < x < nx + hx and -hy < y < ny + hy:
            partial.append(section)
    if partial:
        raise ValueError(
            f"the {width_arcsec:.0f}\" x {height_arcsec:.0f}\" patch at ({ra}, {dec}) crosses the boundary of the GOODS tiles {partial}: "
            "move its centre or make it smaller")
    raise ValueError(f"({ra}, {dec}) is outside the {field} ACS mosaics")


def fetch_goods_cutout(ra, dec, width_arcsec, height_arcsec, band="f606w", field="goods-s", cache_dir=None,
                       kind="drz", max_download_mb=1500.0):
    """Cutout of a GOODS ACS mosaic centred on (ra, dec).

    Returns ``(data, wcs, zp_ab, header)`` like ``candels_cutout.fetch_candels_cutout``: `data` is (ny, nx), in e-/s for ``kind="drz"``
    (the science image) or the inverse-variance weights for ``"wht"`` (``zp_ab`` is then NaN); the pixel grid is the mosaic's, 30 mas, north up.
    """
    field, band = _check(field, band, kind)
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"goods_{FIELDS[field]}_{band}{'' if kind == 'drz' else '_' + kind}"
                                       f"_{ra:.5f}_{dec:+.5f}_{width_arcsec:.1f}x{height_arcsec:.1f}.fits")
    if path is not None and os.path.exists(path):
        with fits.open(path) as hdul:
            data, header = np.array(hdul[0].data, dtype=float), hdul[0].header.copy()
        return data, cc._celestial_wcs(header), float(header.get("ZPAB", float("nan"))), header

    section = find_tile(field, band, ra, dec, width_arcsec, height_arcsec)
    url = tile_url(field, band, section, kind)
    data, out, _ = cc.cutout_from_url(url, ra, dec, width_arcsec, height_arcsec, label=f"the GOODS {field} {band} tile {section}",
                                      max_download_mb=max_download_mb, warn_empty=(kind == "drz"))
    zp_ab = ZP_AB[band] if kind == "drz" else float("nan")
    hdr = out.to_header()
    hdr["BAND"] = band.upper()
    hdr["BUNIT"] = "ELECTRONS/S" if kind == "drz" else ""
    if np.isfinite(zp_ab):                         # a FITS header cannot hold a NaN: no keyword means no zero point
        hdr["ZPAB"] = (float(zp_ab), "AB zero point: m = ZPAB - 2.5 log10(e-/s)")
    hdr["SOURCE"] = url.rsplit("/", 1)[1]
    hdr["FIELD"] = field
    if path is not None:
        fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)
    return data.astype(float), cc._celestial_wcs(hdr), float(zp_ab), hdr
