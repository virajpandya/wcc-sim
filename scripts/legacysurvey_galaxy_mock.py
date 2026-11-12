"""Prototype: DESI Legacy Surveys Tractor galaxies -> mock WCC field.

Run:  python scripts/legacysurvey_galaxy_mock.py [--ra 150.12 --dec 2.2] [--quick]
(network needed on first run for the Tractor and Gaia queries; both are cached)

Queries the Tractor catalog (NOIRLab Astro Data Lab, ``ls_dr10.tractor`` by
default) for the galaxies in one WCC frame, turns every resolved source into a
:class:`wcc_sim.SersicComponent`, and simulates the frame with the real Gaia
stars underneath (``simulate_field(extended_sources=...)``). With ``--plot`` it
also fetches the Legacy Surveys r-band cutout and shows it beside the mock.

Mapping of Tractor columns (see :func:`tractor_to_sersic`):

    type EXP / REX -> n = 1,  DEV -> n = 4,  SER -> n = ``sersic``
    shape_r        -> r_eff_arcsec (half-light radius)
    shape_e1/e2    -> ellip = 1 - q with q = (1 - e) / (1 + e), e = |(e1, e2)|;
                      position angle (E of N) = 0.5 * atan2(e2, e1), checked
                      against LS-DR10 cutout moments for 30 COSMOS galaxies
                      (median |dPA| = 0.8 deg), then carried onto the detector
                      through the frame WCS
    flux_g/r/z     -> AB magnitude in the WCC band -> Gaia-G-equivalent
                      ``total_mag`` that gives the same e-/s through ``template``

Not handled (prototype limits): Tractor ``PSF`` rows are dropped, which also
drops faint compact galaxies and quasars that Gaia does not list; there is no
population below the Tractor depth (r ~ 23.5-24); each galaxy is one
single-Sersic component however clumpy it really is.
"""

import argparse
import io
import os
import time
import warnings

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

from wcc_sim import SersicComponent, detector_frame, empty_catalog, simulate_field
from wcc_sim.starflux import REF_MAG, rate_for_ab, rate_for_spt

TAP_URL = "https://datalab.noirlab.edu/tap"
COSMOS = (150.12, 2.20)

#: Tractor ``type`` -> Sersic index; None means "use the fitted ``sersic`` column".
TYPE_TO_N = {"EXP": 1.0, "REX": 1.0, "DEV": 4.0, "SER": None}

#: Largest ellipticity passed on (a thin edge-on disk, q = 0.05).
MAX_ELLIP = 0.95
N_RANGE = (0.3, 6.2)

#: LS bands used for each WCC filter. ``zwo:i`` has no LS band (DR10 has i only
#: in the south), so it takes the geometric mean of r and z, whose pivots
#: (623, 913 nm) bracket it roughly symmetrically.
BAND_FLUXES = {
    "zwo:g": ("flux_g",),
    "zwo:r": ("flux_r",),
    "zwo:i": ("flux_r", "flux_z"),
    "zwo:z": ("flux_z",),
}

#: Margin beyond the detector half-diagonal so galaxies centred just off the
#: frame still contribute their outskirts [arcsec].
QUERY_MARGIN_ARCSEC = 20.0


def _tractor_adql(table, ra, dec, radius_arcsec):
    """Box query (Data Lab's TAP rejects ADQL POINT/CIRCLE) on primary sources."""
    dec_lo, dec_hi = dec - radius_arcsec / 3600.0, dec + radius_arcsec / 3600.0
    dra = radius_arcsec / 3600.0 / max(np.cos(np.radians(min(abs(dec) + radius_arcsec / 3600.0, 89.0))), 1e-3)
    lo, hi = ra - dra, ra + dra
    if lo < 0.0:
        ra_clause = f"(ra >= {lo + 360.0:.7f} OR ra <= {hi:.7f})"
    elif hi > 360.0:
        ra_clause = f"(ra >= {lo:.7f} OR ra <= {hi - 360.0:.7f})"
    else:
        ra_clause = f"ra BETWEEN {lo:.7f} AND {hi:.7f}"
    return (
        "SELECT ra, dec, type, shape_r, shape_e1, shape_e2, sersic, "
        "flux_g, flux_r, flux_z, mw_transmission_r "
        f"FROM {table} WHERE {ra_clause} "
        f"AND dec BETWEEN {dec_lo:.7f} AND {dec_hi:.7f} AND brick_primary = 1"
    )


#: Legacy Surveys viewer catalog service: a second, independent route to the same
#: Tractor rows when Data Lab is down. ``{layer}`` is the release with dashes.
VIEWER_URL = "https://www.legacysurvey.org/viewer/{layer}/cat.fits"

TRACTOR_COLUMNS = ["ra", "dec", "type", "shape_r", "shape_e1", "shape_e2", "sersic",
                   "flux_g", "flux_r", "flux_z", "mw_transmission_r"]


class _TimeoutSession(requests.Session):
    """requests session with a default timeout.

    pyvo has no client-side timeout (the ``timeout=`` keyword of ``run_sync`` is
    sent to the server as a TAP parameter and ignored), so a hung gateway
    blocked a run for 10 minutes before a 504 came back.
    """

    def __init__(self, timeout):
        super().__init__()
        self._default_timeout = timeout

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self._default_timeout)
        return super().request(*args, **kwargs)


def _tractor_datalab(release, ra, dec, radius_arcsec, timeout):
    import pyvo

    svc = pyvo.dal.TAPService(TAP_URL, session=_TimeoutSession(timeout))
    result = svc.run_sync(_tractor_adql(f"{release}.tractor", ra, dec, radius_arcsec)).to_table()
    result["type"] = np.char.strip(np.asarray(result["type"]).astype(str))
    return result


def _tractor_viewer(release, ra, dec, radius_arcsec, timeout):
    """The same box from the viewer's cat.fits (brick-primary rows, same columns)."""
    dec_lo, dec_hi = dec - radius_arcsec / 3600.0, dec + radius_arcsec / 3600.0
    dra = radius_arcsec / 3600.0 / max(np.cos(np.radians(min(abs(dec) + radius_arcsec / 3600.0, 89.0))), 1e-3)
    lo, hi = ra - dra, ra + dra
    ranges = [(lo, hi)]
    if lo < 0.0:
        ranges = [(lo + 360.0, 360.0), (0.0, hi)]
    elif hi > 360.0:
        ranges = [(lo, 360.0), (0.0, hi - 360.0)]
    url = VIEWER_URL.format(layer=release.replace("_", "-"))
    parts = []
    for ra_lo, ra_hi in ranges:
        resp = requests.get(url, params={"ralo": f"{ra_lo:.7f}", "rahi": f"{ra_hi:.7f}",
                                         "declo": f"{dec_lo:.7f}", "dechi": f"{dec_hi:.7f}"},
                            timeout=timeout)
        resp.raise_for_status()
        tab = Table.read(io.BytesIO(resp.content), format="fits")
        if "brick_primary" in tab.colnames:
            tab = tab[np.asarray(tab["brick_primary"], dtype=bool)]
        out = Table()
        for col in TRACTOR_COLUMNS:
            if col == "type":
                out[col] = np.char.strip(np.asarray(tab[col]).astype(str))
            else:
                out[col] = np.asarray(tab[col], dtype=float)
        parts.append(out)
    return vstack(parts) if len(parts) > 1 else parts[0]


def fetch_tractor(ra, dec, radius_arcsec, release="ls_dr10", cache_dir=None,
                  source="auto", timeout=60.0, retries=1, backoff_s=2.0):
    """Tractor rows within a box of half-width `radius_arcsec` around (ra, dec).

    `source` is ``"datalab"`` (NOIRLab Astro Data Lab TAP), ``"viewer"`` (the
    legacysurvey.org catalog service) or ``"auto"``: Data Lab first, then the
    viewer if Data Lab fails. Each attempt has a client-side `timeout` [s] and is
    retried `retries` times after `backoff_s` seconds; the result is cached, and
    ``meta["source"]`` records which service answered. Falling back warns.
    """
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(
            cache_dir, f"tractor_{release}_{ra:.5f}_{dec:+.5f}_{radius_arcsec:.1f}.ecsv")
        if os.path.exists(path):
            return Table.read(path, format="ascii.ecsv")
    order = {"auto": ("datalab", "viewer"), "datalab": ("datalab",), "viewer": ("viewer",)}
    if source not in order:
        raise ValueError(f"source must be one of {sorted(order)}, got {source!r}")
    fetchers = {"datalab": _tractor_datalab, "viewer": _tractor_viewer}
    errors, result, used = [], None, None
    for name in order[source]:
        for attempt in range(1 + int(retries)):
            try:
                result = fetchers[name](release, ra, dec, radius_arcsec, timeout)
                used = name
                break
            except Exception as err:    # network, gateway, parsing: try again, then the next service
                errors.append(f"{name} (attempt {attempt + 1}): {type(err).__name__}: {str(err)[:150]}")
                if attempt < int(retries):
                    time.sleep(backoff_s)
        if result is not None:
            break
    if result is None:
        raise RuntimeError("could not fetch the Tractor catalog:\n  " + "\n  ".join(errors))
    if errors:
        warnings.warn(f"Tractor catalog came from {used} after: " + "; ".join(errors), UserWarning)
    result.meta["source"] = used
    if path is not None:
        result.write(path, format="ascii.ecsv", overwrite=True)
    return result


def ab_mag_in_band(table, sensorfilter):
    """AB magnitude (observed, i.e. including Milky Way dust) in the WCC band.

    Tractor fluxes are nanomaggies (AB = 22.5 - 2.5 log10 f) and are *not*
    corrected for Galactic extinction, which is what the telescope sees, so
    ``mw_transmission`` is left alone and the components carry ``ebv = 0``.
    """
    if sensorfilter not in BAND_FLUXES:
        raise ValueError(
            f"no Legacy Surveys band mapping for {sensorfilter!r}; "
            f"supported: {sorted(BAND_FLUXES)}")
    flux = np.ones(len(table))
    cols = BAND_FLUXES[sensorfilter]
    for col in cols:
        flux = flux * np.clip(np.asarray(table[col], dtype=float), 0.0, None)
    flux = flux ** (1.0 / len(cols))
    with np.errstate(divide="ignore"):
        return np.where(flux > 0.0, 22.5 - 2.5 * np.log10(flux), np.inf)


def pixel_position_angle(frame, ra, dec, pa_east_of_north):
    """Major-axis angle CCW from the detector +x axis, from sky PA (E of N).

    Moves 1 arcsec along the sky position angle and reads the displacement in
    detector pixels, so any detector rotation (``pa=``) is handled by the WCS.
    """
    here = SkyCoord(ra * u.deg, dec * u.deg)
    there = here.directional_offset_by(pa_east_of_north * u.deg, 1.0 * u.arcsec)
    x0, y0 = frame.wcs.world_to_pixel(here)
    x1, y1 = frame.wcs.world_to_pixel(there)
    return np.degrees(np.arctan2(np.asarray(y1) - y0, np.asarray(x1) - x0)) % 180.0


def tractor_to_sersic(table, frame, sensorfilter, template="G2V", max_mag=None):
    """One SersicComponent per resolved Tractor source; also returns a truth table.

    `max_mag` drops sources fainter than this AB magnitude in the WCC band.
    """
    if len(table) == 0:
        return [], Table()
    types = np.char.strip(np.asarray(table["type"]).astype(str))
    r_eff = np.asarray(table["shape_r"], dtype=float)
    keep = np.isin(types, list(TYPE_TO_N)) & np.isfinite(r_eff) & (r_eff > 0.0)
    m_ab = ab_mag_in_band(table, sensorfilter)
    keep &= np.isfinite(m_ab)
    if max_mag is not None:
        keep &= m_ab <= max_mag
    t = table[keep]
    types, r_eff, m_ab = types[keep], r_eff[keep], m_ab[keep]

    e1 = np.asarray(t["shape_e1"], dtype=float)
    e2 = np.asarray(t["shape_e2"], dtype=float)
    e = np.hypot(e1, e2)
    q = (1.0 - e) / (1.0 + e)
    ellip = np.clip(1.0 - q, 0.0, MAX_ELLIP)
    pa_sky = (0.5 * np.degrees(np.arctan2(e2, e1))) % 180.0
    ra = np.asarray(t["ra"], dtype=float)
    dec = np.asarray(t["dec"], dtype=float)
    pa_pix = pixel_position_angle(frame, ra, dec, pa_sky)

    n = np.array([TYPE_TO_N[ty] if TYPE_TO_N[ty] is not None else np.nan for ty in types])
    ser = np.isnan(n)
    n[ser] = np.asarray(t["sersic"], dtype=float)[ser]
    n = np.clip(n, *N_RANGE)

    # Same e-/s as an AB-flat source of m_ab, expressed as the Gaia-G Vega
    # magnitude SersicComponent wants for this template.
    rate_spt = rate_for_spt(template, sensorfilter)
    g_equiv = REF_MAG - 2.5 * np.log10(
        np.array([rate_for_ab(m, sensorfilter) for m in m_ab]) / rate_spt)

    comps = [
        SersicComponent(ra=float(ra[i]), dec=float(dec[i]), n=float(n[i]),
                        r_eff_arcsec=float(r_eff[i]), ellip=float(ellip[i]),
                        pa_deg=float(pa_pix[i]), total_mag=float(g_equiv[i]),
                        template=template)
        for i in range(len(t))
    ]
    truth = Table({
        "ra": ra, "dec": dec, "type": types, "n": n, "r_eff_arcsec": r_eff,
        "ellip": ellip, "pa_sky_deg": pa_sky, "pa_pix_deg": pa_pix,
        "mag_ab_band": m_ab, "total_mag_G": g_equiv,
    })
    return comps, truth


def fetch_ls_cutout(ra, dec, size_arcsec, layer="ls-dr10", band="r", npix=512):
    """Legacy Surveys FITS cutout (viewer service): (data, wcs)."""
    import requests
    from astropy.io import fits
    from astropy.wcs import WCS

    pixscale = size_arcsec / npix
    url = (f"https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}"
           f"&layer={layer}&pixscale={pixscale:.5f}&size={npix}&bands={band}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    hdu = fits.open(io.BytesIO(resp.content))[0]
    data = hdu.data[0] if hdu.data.ndim == 3 else hdu.data
    return np.asarray(data, dtype=float), WCS(hdu.header, naxis=2)


def plot_comparison(field, comps, ra, dec, band, path, bin_factor=16):
    """LS cutout beside the mock frame, both on one coarse pixel grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reproject import reproject_interp

    ny, nx = field.image_e.shape
    cy, cx = ny // bin_factor, nx // bin_factor
    # WCS slicing with a step ignores the CD matrix, so rescale it by hand:
    # coarse pixel j covers original pixels (j-1)*b+1 .. j*b (FITS 1-based).
    coarse_wcs = field.wcs.deepcopy()
    coarse_wcs.wcs.cd = field.wcs.wcs.cd * bin_factor
    coarse_wcs.wcs.crpix = (field.wcs.wcs.crpix - 0.5) / bin_factor + 0.5
    coarse_wcs.array_shape = (cy, cx)
    binned = field.image_clean[:cy * bin_factor, :cx * bin_factor].reshape(
        cy, bin_factor, cx, bin_factor).sum(axis=(1, 3))

    pix_arcsec = field.params.get("plate_scale_mas", 16.87) / 1000.0
    size = 1.1 * max(nx, ny) * pix_arcsec
    ls, ls_wcs = fetch_ls_cutout(ra, dec, size, band=band)
    ls_on_grid, _ = reproject_interp((ls, ls_wcs), coarse_wcs, shape_out=(cy, cx))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, img, title in ((axes[0], ls_on_grid, f"LS-DR10 {band}-band cutout"),
                           (axes[1], binned, f"mock WCC ({len(comps)} Sersic galaxies + Gaia)")):
        good = np.isfinite(img)
        lo, hi = np.nanpercentile(img[good], [5, 99.7])
        ax.imshow(np.arcsinh((img - lo) / max(hi - lo, 1e-12) * 10.0), origin="lower",
                  cmap="gray", vmin=0, vmax=np.arcsinh(10.0))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ra", type=float, default=COSMOS[0])
    ap.add_argument("--dec", type=float, default=COSMOS[1])
    ap.add_argument("--sensorfilter", default="zwo:r")
    ap.add_argument("--release", default="ls_dr10", help="Data Lab schema, e.g. ls_dr9")
    ap.add_argument("--exptime", type=float, default=300.0)
    ap.add_argument("--pa", type=float, default=0.0, help="detector rotation, deg E of N")
    ap.add_argument("--max-mag", type=float, default=None,
                    help="drop galaxies fainter than this AB mag in the WCC band")
    ap.add_argument("--quick", action="store_true", help="1024x1024 subarray")
    ap.add_argument("--no-gaia", action="store_true", help="galaxies only")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cache-dir", default="ls_mock_cache")
    ap.add_argument("--out-dir", default="ls_mock_out")
    ap.add_argument("--plot", action="store_true", help="also fetch the LS cutout and compare")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    shape = (1024, 1024) if args.quick else None
    frame = detector_frame(args.sensorfilter, (args.ra, args.dec), pa=args.pa, shape=shape)
    ny, nx = frame.shape
    half_diag = 0.5 * np.hypot(nx, ny) * frame.plate_scale_mas / 1000.0

    t0 = time.time()
    tractor = fetch_tractor(args.ra, args.dec, half_diag + QUERY_MARGIN_ARCSEC,
                            release=args.release, cache_dir=args.cache_dir)
    comps, truth = tractor_to_sersic(tractor, frame, args.sensorfilter, max_mag=args.max_mag)
    kinds = dict(zip(*np.unique(np.asarray(tractor["type"]).astype(str), return_counts=True)))
    print(f"Tractor rows {len(tractor)} {kinds} -> {len(comps)} Sersic components "
          f"({time.time() - t0:.1f} s)")
    if len(comps) == 0:
        raise SystemExit("no resolved Tractor sources in this field (outside the footprint?)")
    truth.write(os.path.join(args.out_dir, "tractor_components.ecsv"),
                format="ascii.ecsv", overwrite=True)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        field = simulate_field(
            args.ra, args.dec, sensorfilter=args.sensorfilter, exptime=args.exptime,
            pa=args.pa, shape=shape, seed=args.seed, extended_sources=comps,
            catalog=empty_catalog() if args.no_gaia else None,
            cache_dir=os.path.join(args.cache_dir, "gaia"),
        )
    print(f"simulated {nx}x{ny} px in {time.time() - t0:.1f} s; "
          f"{len(field.catalog)} Gaia stars")
    field.write(os.path.join(args.out_dir, "mock_wcc.fits"))
    if args.plot:
        band = args.sensorfilter.split(":")[1]
        band = band if band in ("g", "r", "z") else "r"
        plot_comparison(field, comps, args.ra, args.dec, band,
                        os.path.join(args.out_dir, "mock_vs_legacysurvey.png"))
    print(f"wrote {args.out_dir}/")


if __name__ == "__main__":
    main()
