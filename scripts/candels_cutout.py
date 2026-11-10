"""Small cutouts from the CANDELS HST mosaics, without downloading them.

The mosaics (``*-tot``, v1.0, MAST HLSP) are 0.8-8 GB each, but a FITS image
is one contiguous block of rows, so the rows a cutout needs can be read with a
single HTTP byte-range request and sliced in memory. The result is cached as a
small FITS file with a WCS.

    from candels_cutout import fetch_candels_cutout, mosaic_info
    data, wcs, zp_ab, header = fetch_candels_cutout(150.12, 2.20, 52.0, 35.0, "f606w",
                                                    cache_dir="ls_cache")           # COSMOS
    data, wcs, zp_ab, header = fetch_candels_cutout(215.005, 52.935, 52.0, 35.0, "f606w",
                                                    field="egs")
    mosaic_info("egs", "f606w")           # shape, pixel scale, centre and rotation, for choosing a patch

Fields and bands (``list_fields()``):

    cosmos, uds, egs   ACS f606w f814w (30 mas/px) and WFC3 f125w f160w (60 mas/px)
    goods-n, goods-s   the same plus ACS f850lp and WFC3 f105w

In EGS and GOODS-N the 30 mas ACS mosaics and the 60 mas ones are separate files;
``acs_scale="60mas"`` selects the coarser ACS pair (a quarter of the data).

The transfer is rows x mosaic width x 4 bytes, not the cutout size: about 130 MB
for the 52" x 35" COSMOS patch (28000 px wide), 375 MB in EGS (81600 px wide).
Measured: 13-300 s for COSMOS (the archive's speed varies 20x between runs), 32 s for
EGS. ``max_download_mb`` stops a request that would pull more than that.
The EGS mosaic is rotated against north (see ``mosaic_info(...)["rotation_deg"]`` and
:func:`bounding_cutout_size`): a cutout is a rectangle in the mosaic's own pixels, so to cover a
north-up frame it has to be larger than the frame.
A position outside a field's footprint raises ValueError; a patch that is mostly
empty (inside the bounding box but outside the covered area) warns.

Data are in e-/s; ``zp_ab`` converts a pixel value v to AB magnitude as
``zp_ab - 2.5 log10(v)`` (see :func:`wcc_sim.phangs.ab_zeropoint`).
"""

import os
import re
import time
import warnings

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

ROOT = "https://archive.stsci.edu/pub/hlsp/candels/"

_ACS_BANDS = ("f606w", "f814w", "f850lp")
_COMMON = ("f606w", "f814w", "f125w", "f160w")
_GOODS = ("f606w", "f814w", "f850lp", "f105w", "f125w", "f160w")

#: field -> directory under ROOT, short tag for cache names, whether ACS and WFC3
#: live in separate 30mas-acs / 60mas-all folders, and the available bands.
FIELDS = {
    "cosmos": {"dir": "cosmos/cos-tot", "tag": "cos", "split": False, "bands": _COMMON},
    "uds": {"dir": "uds/uds-tot", "tag": "uds", "split": False, "bands": _COMMON},
    "egs": {"dir": "egs/egs-tot", "tag": "egs", "split": True, "bands": _COMMON},
    "goods-n": {"dir": "goods-n/gn-tot", "tag": "gn", "split": True, "bands": _GOODS},
    "goods-s": {"dir": "goods-s/gs-tot", "tag": "gs", "split": False, "bands": _GOODS},
}
_KINDS = ("drz", "exp", "wht", "rms")
_BLOCK = 2880
_TIMEOUT = 600

#: Header keywords of the distortion models of the input frames. The drizzled mosaics
#: have the distortion removed (CTYPE is a plain TAN), but the SIP terms and the
#: lookup-table keywords are still in the header: astropy would apply the first and
#: refuses to build a WCS from the second without the file object (EGS).
_DISTORTION = re.compile(r"^(?:(?:A|B|AP|BP)_(?:ORDER|\d+_\d+)|CPDIS\d*|CQDIS\d*|CPERR\d*|CQERR\d*"
                         r"|D2IM\w*|DP[12](?:\..*)?)$")


def list_fields():
    """{field: bands} of the supported CANDELS fields."""
    return {name: info["bands"] for name, info in FIELDS.items()}


def mosaic_url(field, band, kind="drz", acs_scale="30mas"):
    """URL of the CANDELS total mosaic (``kind`` = drz, exp, wht or rms) of one band."""
    field, band = str(field).lower(), str(band).lower()
    if field not in FIELDS:
        raise ValueError(f"field must be one of {sorted(FIELDS)}, got {field!r}")
    info = FIELDS[field]
    if band not in info["bands"]:
        raise ValueError(f"{field} has no {band} mosaic; bands: {list(info['bands'])}")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {list(_KINDS)}, got {kind!r}")
    if acs_scale not in ("30mas", "60mas"):
        raise ValueError(f"acs_scale must be '30mas' or '60mas', got {acs_scale!r}")
    instrument = "acs" if band in _ACS_BANDS else "wfc3"
    mosaic = info["dir"].split("/")[1]                      # cos-tot, egs-tot, gn-tot, ...
    if info["split"]:
        scale = acs_scale if instrument == "acs" else "60mas"
        folder = "30mas-acs/" if scale == "30mas" else "60mas-all/"
        stem = f"{instrument}_{mosaic}-{scale}"
    else:
        folder, stem = "", f"{instrument}_{mosaic}"
    return f"{ROOT}{info['dir']}/v1.0/{folder}hlsp_candels_hst_{stem}_{band}_v1.0_{kind}.fits"


def _get(url, headers=None, attempts=5, backoff_s=3.0):
    """requests.get that retries connection errors, timeouts and HTTP 5xx.

    The archive sometimes refuses new connections for a few seconds; without a retry
    that ends a run at the first request. 4xx answers (a missing file) are not retried.
    """
    for k in range(attempts):
        last = k == attempts - 1
        try:
            r = requests.get(url, headers=headers, timeout=_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout):
            if last:
                raise
        else:
            if r.status_code < 500 or last:
                return r
        time.sleep(backoff_s * (k + 1))


def _read_primary_header(url):
    """(header, byte offset of the data) from the first few KB of the file."""
    size = 20 * _BLOCK
    while True:
        r = _get(url, headers={"Range": f"bytes=0-{size - 1}"})
        if r.status_code == 404:
            raise ValueError(f"no such mosaic on the archive: {url}")
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
    """Celestial WCS of a drizzled mosaic, without the leftover distortion keywords."""
    hd = header.copy()
    for key in [k for k in hd.keys() if _DISTORTION.match(k)]:
        del hd[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(hd, naxis=2)
    wcs.sip = None
    return wcs


def _pixel_scale_arcsec(wcs):
    return float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600.0)


def mosaic_info(field, band="f606w", kind="drz", acs_scale="30mas"):
    """Shape, pixel scale and centre of a mosaic from its header (a fraction of a second)."""
    url = mosaic_url(field, band, kind, acs_scale)
    header, _ = _read_primary_header(url)
    wcs = _celestial_wcs(header)
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    ra, dec = wcs.pixel_to_world_values(nx / 2.0, ny / 2.0)
    m = wcs.pixel_scale_matrix                  # sky displacement (east, north) per pixel step
    rotation = float(np.degrees(np.arctan2(m[0, 1], m[1, 1])))
    return {"url": url, "nx": nx, "ny": ny, "pixel_scale_arcsec": _pixel_scale_arcsec(wcs),
            "centre_ra": float(ra), "centre_dec": float(dec), "size_gb": nx * ny * 4 / 1e9,
            "rotation_deg": rotation}


def bounding_cutout_size(width_arcsec, height_arcsec, rotation_deg):
    """(width, height) of a cutout, in the mosaic's pixel axes, that contains a north-up
    rectangle of the given size.

    ``mosaic_info(...)["rotation_deg"]`` is the angle of the mosaic's +y axis east of
    north. COSMOS, UDS and GOODS are north-up (0), but the EGS mosaic is rotated, and a
    cutout of the frame's own size would leave the corners of a north-up frame without data.
    """
    c, s = abs(np.cos(np.radians(rotation_deg))), abs(np.sin(np.radians(rotation_deg)))
    return width_arcsec * c + height_arcsec * s, width_arcsec * s + height_arcsec * c


def cutout_from_url(url, ra, dec, width_arcsec, height_arcsec, label="mosaic", max_download_mb=1500.0,
                    warn_empty=True):
    """Pixel-aligned cutout of a 32-bit FITS mosaic by one HTTP byte-range read.

    A FITS image is one contiguous block of rows, so the rows of the cutout are read whole
    (rows x mosaic width x 4 bytes) and sliced in memory. Returns ``(data, wcs, header)``:
    the float32 cutout, its celestial WCS, and the header of the mosaic. The cutout size is
    ``round(size / pixel scale)`` pixels, centred on (ra, dec) to a pixel. `label` names the
    mosaic in error messages.
    """
    header, offset = _read_primary_header(url)
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    if header["BITPIX"] != -32:
        raise ValueError(f"unexpected BITPIX {header['BITPIX']} in {url}")
    wcs = _celestial_wcs(header)
    scale = _pixel_scale_arcsec(wcs)
    xc, yc = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
    if not (np.isfinite(xc) and np.isfinite(yc)):
        raise ValueError(f"({ra}, {dec}) is far from {label} (it does not project onto the mosaic)")
    npx, npy = max(1, int(round(width_arcsec / scale))), max(1, int(round(height_arcsec / scale)))
    x0, y0 = int(round(xc - npx / 2.0)), int(round(yc - npy / 2.0))     # fixed size: no +-1 px from rounding noise
    x1, y1 = x0 + npx, y0 + npy
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        raise ValueError(
            f"({ra}, {dec}) with a {width_arcsec}\" x {height_arcsec}\" patch is outside "
            f"{label} (x {x0}..{x1} of {nx}, y {y0}..{y1} of {ny})")

    row_bytes = nx * 4
    transfer_mb = (y1 - y0) * row_bytes / 1e6
    if transfer_mb > max_download_mb:
        raise ValueError(
            f"this patch needs {transfer_mb:.0f} MB of whole mosaic rows ({nx} px wide), above "
            f"max_download_mb={max_download_mb:.0f}; use a smaller patch, acs_scale='60mas', or raise the limit")
    first, last = offset + y0 * row_bytes, offset + y1 * row_bytes - 1
    r = _get(url, headers={"Range": f"bytes={first}-{last}"})
    r.raise_for_status()
    if r.status_code != 206 or len(r.content) != last - first + 1:
        raise RuntimeError(f"byte-range read failed (status {r.status_code}, {len(r.content)} bytes)")
    band_rows = np.frombuffer(r.content, dtype=">f4").reshape(y1 - y0, nx)
    data = band_rows[:, x0:x1].astype(np.float32)
    if warn_empty:
        empty = float(np.mean(~np.isfinite(data) | (data == 0.0)))
        if empty > 0.2:
            warnings.warn(f"{empty:.0%} of this patch is empty (no data): it lies inside "
                          f"{label} but largely outside the area it covers", UserWarning)
    out = _celestial_wcs(header)
    out.wcs.crpix = out.wcs.crpix - [x0, y0]
    return data, out, header


def fetch_candels_cutout(ra, dec, width_arcsec, height_arcsec, band="f606w", cache_dir=None,
                         kind="drz", field="cosmos", acs_scale="30mas", max_download_mb=1500.0):
    """Cutout of a CANDELS mosaic centred on (ra, dec).

    Returns ``(data, wcs, zp_ab, header)``: `data` is (ny, nx) in e-/s with the
    pixel grid of the mosaic (north up), `wcs` its celestial WCS.

    `kind` picks the product on the same grid: ``"drz"`` the science image,
    ``"exp"`` the exposure-time map in seconds (``zp_ab`` is then NaN),
    ``"wht"`` / ``"rms"`` the weight and noise maps. `field` and `acs_scale` are
    described in the module docstring.
    """
    field, band = str(field).lower(), str(band).lower()
    url = mosaic_url(field, band, kind, acs_scale)              # validates field, band, kind
    info = FIELDS[field]
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tag = info["tag"] + ("-60mas" if info["split"] and band in _ACS_BANDS and acs_scale == "60mas" else "")
        path = os.path.join(
            cache_dir,
            f"candels_{tag}_{band}{'' if kind == 'drz' else '_' + kind}_{ra:.5f}_{dec:+.5f}"
            f"_{width_arcsec:.1f}x{height_arcsec:.1f}.fits")
    if path is not None and os.path.exists(path):
        with fits.open(path) as hdul:
            data, header = np.array(hdul[0].data, dtype=float), hdul[0].header.copy()
        return data, _celestial_wcs(header), float(header.get("ZPAB", float("nan"))), header

    data, out, header = cutout_from_url(url, ra, dec, width_arcsec, height_arcsec,
                                        label=f"the CANDELS {field} {band} mosaic", max_download_mb=max_download_mb,
                                        warn_empty=(kind == "drz"))
    if "PHOTFLAM" in header and "PHOTPLAM" in header:
        zp_ab = -2.5 * np.log10(header["PHOTFLAM"]) - 5.0 * np.log10(header["PHOTPLAM"]) - 2.408
    else:
        zp_ab = float("nan")
    hdr = out.to_header()
    hdr["BAND"] = band.upper()
    hdr["BUNIT"] = {"drz": "ELECTRONS/S", "exp": "s"}.get(kind, "")
    if np.isfinite(zp_ab):                         # a FITS header cannot hold a NaN: no keyword means no zero point
        hdr["ZPAB"] = (float(zp_ab), "AB zero point: m = ZPAB - 2.5 log10(e-/s)")
    hdr["SOURCE"] = url.rsplit("/", 1)[1]
    hdr["FIELD"] = field
    if path is not None:
        fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)
    return data.astype(float), _celestial_wcs(hdr), float(zp_ab), hdr
