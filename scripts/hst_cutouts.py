"""Small cutouts from the HST mosaics of the CANDELS, UVCANDELS and GOODS surveys, without downloading them.

    from hst_cutouts import fetch_candels_cutout, fetch_goods_cutout, mosaic_info
    data, wcs, zp_ab, header = fetch_candels_cutout(150.12, 2.20, 52.0, 35.0, "f606w", cache_dir="ls_cache")      # COSMOS
    data, wcs, zp_ab, header = fetch_goods_cutout(53.16, -27.82, 180.0, 180.0, "f435w", field="goods-s")          # the original GOODS ACS survey

The mosaics are 0.8-8 GB each, but a FITS image is one contiguous block of rows, so the rows a cutout needs are read with a single HTTP byte-range
request (:func:`cutout_from_url`) and sliced in memory. The result is cached as a small FITS file with a WCS. Data are in e-/s; ``zp_ab`` converts a pixel value v
to AB magnitude as ``zp_ab - 2.5 log10(v)``.

CANDELS (``fetch_candels_cutout``; MAST HLSP ``candels``, ``*-tot`` mosaics, v1.0), ``list_fields()``:

    cosmos, uds, egs   ACS f606w f814w (30 mas/px) and WFC3 f125w f160w (60 mas/px)
    cosmos, egs        also ACS f435w, from UVCANDELS (central ~30% of the mosaic; no exposure map)
    goods-n, goods-s   the same CANDELS bands plus ACS f850lp and WFC3 f105w (no f435w in CANDELS)

In EGS and GOODS-N the 30 mas ACS mosaics and the 60 mas ones are separate files; ``acs_scale="60mas"`` selects the coarser ACS pair (a quarter of the data).
The transfer is rows x mosaic width x 4 bytes, not the cutout size: about 130 MB for the 52" x 35" COSMOS patch (28000 px wide), 375 MB in EGS (81600 px wide), 1 GB
for a 3' x 3' patch of GOODS-South (64800 px wide). ``max_download_mb`` stops a request that would pull more than that. The EGS mosaic is rotated against north
(``mosaic_info(...)["rotation_deg"]``, :func:`bounding_cutout_size`): a cutout is a rectangle in the mosaic's own pixels, so to cover a north-up frame it has to be larger.
A position outside a field's footprint raises ValueError; a patch that is mostly empty (inside the bounding box but outside the covered area) warns.

GOODS (``fetch_goods_cutout``; MAST HLSP ``goods/v2``, Giavalisco et al. 2004): ACS **F435W (B), F606W (V), F775W (i) and F850LP (z)** over GOODS-North and GOODS-South at
30 mas/pixel, the blue band that CANDELS lacks. The tiles ("sections") are 8192 x 8192 pixels on one common grid; a cutout is read from the one tile that contains it, and a
patch that crosses a tile boundary raises ValueError (move its centre or make it smaller). The tiles have no exposure map: ``GOODS_TYPICAL_EXPOSURE_S`` gives the net exposure
of a typical region, from the GOODS v2.0 documentation, and ``GOODS_ZP_AB`` the zero points (the PHOTFLAM keywords of these headers are zero).

:func:`mask_to_sources` cleans an image before it goes into ``wcc_sim.ImageComponent``.
"""

import os
import re
import time
import warnings

import functools

import numpy as np
import requests
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

ROOT = "https://archive.stsci.edu/pub/hlsp/candels/"

#: UVCANDELS (Mehta et al. 2024) added ACS F435W over the central part of COSMOS and EGS, on the CANDELS pixel grid.
UVCANDELS_ROOT = "https://archive.stsci.edu/hlsps/uvcandels/"

_ACS_BANDS = ("f435w", "f606w", "f814w", "f850lp")
_COMMON = ("f606w", "f814w", "f125w", "f160w")
_UV = _COMMON + ("f435w",)
_GOODS = ("f606w", "f814w", "f850lp", "f105w", "f125w", "f160w")

#: field -> directory under ROOT, short tag for cache names, whether ACS and WFC3
#: live in separate 30mas-acs / 60mas-all folders, and the available bands.
FIELDS = {
    "cosmos": {"dir": "cosmos/cos-tot", "tag": "cos", "split": False, "bands": _UV},
    "uds": {"dir": "uds/uds-tot", "tag": "uds", "split": False, "bands": _COMMON},
    "egs": {"dir": "egs/egs-tot", "tag": "egs", "split": True, "bands": _UV},
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
    if band == "f435w":                                    # UVCANDELS: its own archive tree, both scales for both fields
        if kind == "exp":
            raise ValueError("the UVCANDELS F435W mosaics have no exposure map (kind 'exp'); use 'wht' or 'rms'")
        product = {"drz": "sci", "wht": "wht", "rms": "rms"}[kind]
        return f"{UVCANDELS_ROOT}{field}/hlsp_uvcandels_hst_acs-wfc_{field}-{acs_scale}_f435w_v1.0_{product}.fits"
    instrument = "acs" if band in _ACS_BANDS else "wfc3"
    mosaic = info["dir"].split("/")[1]                      # cos-tot, egs-tot, gn-tot, ...
    if info["split"]:
        scale = acs_scale if instrument == "acs" else "60mas"
        folder = "30mas-acs/" if scale == "30mas" else "60mas-all/"
        stem = f"{instrument}_{mosaic}-{scale}"
    else:
        folder, stem = "", f"{instrument}_{mosaic}"
    return f"{ROOT}{info['dir']}/v1.0/{folder}hlsp_candels_hst_{stem}_{band}_v1.0_{kind}.fits"


def get_with_retry(url, headers=None, attempts=5, backoff_s=3.0):
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
        r = get_with_retry(url, headers={"Range": f"bytes=0-{size - 1}"})
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
    r = get_with_retry(url, headers={"Range": f"bytes={first}-{last}"})
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
        tag = info["tag"] + ("-60mas" if (info["split"] or band == "f435w") and band in _ACS_BANDS and acs_scale == "60mas" else "")
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


# ---------------------------------------------------------------- GOODS ACS (original survey)

GOODS_ROOT = "https://archive.stsci.edu/pub/hlsp/goods/v2/"
GOODS_FIELDS = {"goods-n": "n", "goods-s": "s"}
GOODS_BANDS = {"f435w": "b", "f606w": "v", "f775w": "i", "f850lp": "z"}

#: AB zero points of the v2.0 science images (e-/s), from the GOODS v2.0 documentation.
GOODS_ZP_AB = {"f435w": 25.673, "f606w": 26.486, "f775w": 25.654, "f850lp": 24.862}

#: Net exposure [s] in a typical region of the v2.0 mosaics (GOODS v2.0 documentation).
GOODS_TYPICAL_EXPOSURE_S = {
    "goods-n": {"f435w": 7200.0, "f606w": 5650.0, "f775w": 8530.0, "f850lp": 24760.0},
    "goods-s": {"f435w": 7200.0, "f606w": 5450.0, "f775w": 7028.0, "f850lp": 18232.0},
}
_GOODS_KINDS = {"drz": "drz", "wht": "wht"}


def _goods_check(field, band, kind="drz"):
    field, band = str(field).lower(), str(band).lower()
    if field not in GOODS_FIELDS:
        raise ValueError(f"field must be one of {sorted(GOODS_FIELDS)}, got {field!r}")
    if band not in GOODS_BANDS:
        raise ValueError(f"band must be one of {sorted(GOODS_BANDS)}, got {band!r}")
    if kind not in _GOODS_KINDS:
        raise ValueError(f"kind must be one of {sorted(_GOODS_KINDS)}, got {kind!r}")
    return field, band


def goods_tile_url(field, band, section, kind="drz"):
    """URL of one tile, e.g. ``section="sect23"``."""
    field, band = _goods_check(field, band, kind)
    return f"{GOODS_ROOT}h_{GOODS_FIELDS[field]}{GOODS_BANDS[band]}_{section}_v2.0_{_GOODS_KINDS[kind]}_img.fits"


@functools.lru_cache(maxsize=None)
def list_goods_sections(field, band):
    """The section names of a field and band (``"sect11"``, ...), from the archive listing."""
    field, band = _goods_check(field, band)
    page = get_with_retry(GOODS_ROOT).text
    pattern = rf'href="h_{GOODS_FIELDS[field]}{GOODS_BANDS[band]}_(sect\d+)_v2\.0_drz_img\.fits"'
    return tuple(sorted(set(re.findall(pattern, page))))


@functools.lru_cache(maxsize=None)
def _goods_tile_geometry(field, band, section):
    header, _ = _read_primary_header(goods_tile_url(field, band, section))
    return int(header["NAXIS1"]), int(header["NAXIS2"]), _celestial_wcs(header)


def find_goods_tile(field, band, ra, dec, width_arcsec, height_arcsec):
    """Section whose tile contains the whole patch centred on (ra, dec)."""
    field, band = _goods_check(field, band)
    partial = []
    for section in list_goods_sections(field, band):
        nx, ny, wcs = _goods_tile_geometry(field, band, section)
        scale = _pixel_scale_arcsec(wcs)
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

    Returns ``(data, wcs, zp_ab, header)`` like :func:`fetch_candels_cutout`: `data` is (ny, nx), in e-/s for ``kind="drz"``
    (the science image) or the inverse-variance weights for ``"wht"`` (``zp_ab`` is then NaN); the pixel grid is the mosaic's, 30 mas, north up.
    """
    field, band = _goods_check(field, band, kind)
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"goods_{GOODS_FIELDS[field]}_{band}{'' if kind == 'drz' else '_' + kind}"
                                       f"_{ra:.5f}_{dec:+.5f}_{width_arcsec:.1f}x{height_arcsec:.1f}.fits")
    if path is not None and os.path.exists(path):
        with fits.open(path) as hdul:
            data, header = np.array(hdul[0].data, dtype=float), hdul[0].header.copy()
        return data, _celestial_wcs(header), float(header.get("ZPAB", float("nan"))), header

    section = find_goods_tile(field, band, ra, dec, width_arcsec, height_arcsec)
    url = goods_tile_url(field, band, section, kind)
    data, out, _ = cutout_from_url(url, ra, dec, width_arcsec, height_arcsec, label=f"the GOODS {field} {band} tile {section}",
                                      max_download_mb=max_download_mb, warn_empty=(kind == "drz"))
    zp_ab = GOODS_ZP_AB[band] if kind == "drz" else float("nan")
    hdr = out.to_header()
    hdr["BAND"] = band.upper()
    hdr["BUNIT"] = "ELECTRONS/S" if kind == "drz" else ""
    if np.isfinite(zp_ab):                         # a FITS header cannot hold a NaN: no keyword means no zero point
        hdr["ZPAB"] = (float(zp_ab), "AB zero point: m = ZPAB - 2.5 log10(e-/s)")
    hdr["SOURCE"] = url.rsplit("/", 1)[1]
    hdr["FIELD"] = field
    if path is not None:
        fits.PrimaryHDU(data, header=hdr).writeto(path, overwrite=True)
    return data.astype(float), _celestial_wcs(hdr), float(zp_ab), hdr


# ---------------------------------------------------------------- preparing an image for ImageComponent


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
