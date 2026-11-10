"""Ancillary data for galaxies of the GOODS fields: DESI spectra and 3D-HST redshifts, masses and star formation rates.

    import goods_ancillary as ga
    spec = ga.desi_galaxy_spectra(189.19, 62.20, 40.0)               # DESI DR1 galaxies within 40" (needs ``pip install sparclclient``)
    wave, flux, ivar, model = ga.desi_spectrum(spec["sparcl_id"][0])
    cat = ga.fetch_3dhst("goods-n", cache_dir="ls_cache")             # z_spec, z_peak, log mass, log SFR of 38000 sources
    idx, sep = ga.match_catalog(spec["ra"], spec["dec"], cat)         # nearest 3D-HST source of each galaxy

DESI spectra come from the NOIRLab SPARCL service. They exist only where DESI has observed: GOODS-North is covered (155 galaxy spectra within
0.16 deg of the Hubble Deep Field North in DR1), GOODS-South (Dec -27.8) is outside the DESI footprint.

The 3D-HST v4.1 catalogs (Skelton et al. 2014; MAST HLSP ``3d-hst``) hold, for every source of the CANDELS WFC3 area: the spectroscopic redshift
``z_spec`` (-1 where there is none), the EAZY photometric redshift ``z_peak``, and the FAST stellar mass ``lmass`` (log10 Msun) and star formation
rate ``lsfr`` (log10 Msun/yr) from SED fits at the best redshift. Masses and SFRs of faint galaxies are SED-fit estimates, uncertain by a factor of a few.
"""

import io
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
import astropy.units as u

from candels_cutout import _get

THREEDHST_BASE = "https://archive.stsci.edu/missions/hlsp/3d-hst/RELEASE_V4.0/Photometry/"
THREEDHST_FIELDS = {"goods-n": ("GOODS-N", "goodsn"), "goods-s": ("GOODS-S", "goodss")}
_PARTS = (("Catalog", "cat"), ("Eazy", "zout"), ("Fast", "fout"))


def threedhst_url(field, part):
    """URL of one 3D-HST v4.1 FITS catalog; `part` is ``"cat"``, ``"zout"`` (EAZY redshifts) or ``"fout"`` (FAST masses, SFRs)."""
    field = str(field).lower()
    if field not in THREEDHST_FIELDS:
        raise ValueError(f"field must be one of {sorted(THREEDHST_FIELDS)}, got {field!r}")
    folder = dict((p, d) for d, p in _PARTS).get(part)
    if folder is None:
        raise ValueError(f"part must be one of {[p for _, p in _PARTS]}, got {part!r}")
    directory, stem = THREEDHST_FIELDS[field]
    return f"{THREEDHST_BASE}{directory}/{stem}_3dhst.v4.1.cats/{folder}/{stem}_3dhst.v4.1.{part}.FITS"


def fetch_3dhst(field="goods-n", cache_dir=None):
    """3D-HST catalog of a field as one table: ``id ra dec z_spec z_peak lmass lsfr star_flag`` (about 40 MB downloaded once)."""
    field = str(field).lower()
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"3dhst_{field}_v4.1_summary.fits")
        if os.path.exists(path):
            return Table.read(path)
    tables = {}
    for _, part in _PARTS:
        tables[part] = Table.read(io.BytesIO(_get(threedhst_url(field, part)).content), format="fits")
    cat = tables["cat"]["id", "ra", "dec", "z_spec", "star_flag"]
    zout = tables["zout"]["id", "z_peak"]
    fout = tables["fout"]["id", "lmass", "lsfr"]
    out = join(join(cat, zout, keys="id"), fout, keys="id")
    out.sort("id")
    if path is not None:
        out.write(path, overwrite=True)
    return out


def match_catalog(ra, dec, table, max_sep_arcsec=1.0):
    """Nearest source of `table` (columns ``ra``, ``dec``) to each position: ``(index, separation_arcsec)``, index -1 beyond `max_sep_arcsec`."""
    sky = SkyCoord(np.atleast_1d(ra) * u.deg, np.atleast_1d(dec) * u.deg)
    cat = SkyCoord(np.asarray(table["ra"], dtype=float) * u.deg, np.asarray(table["dec"], dtype=float) * u.deg)
    idx, sep, _ = sky.match_to_catalog_sky(cat)
    sep = sep.arcsec
    return np.where(sep <= max_sep_arcsec, idx, -1), sep


def _sparcl():
    try:
        from sparcl.client import SparclClient
    except ImportError as err:
        raise ImportError("DESI spectra need the NOIRLab SPARCL client: pip install sparclclient") from err
    return SparclClient()


def desi_galaxy_spectra(ra, dec, radius_arcsec, data_release="DESI-DR1", spectypes=("GALAXY",), limit=5000):
    """DESI spectra within `radius_arcsec` of (ra, dec), as a table sorted by distance.

    Columns: ``sparcl_id targetid ra dec redshift redshift_err spectype sep_arcsec``. A target observed more than once is listed once.
    """
    half_dec = radius_arcsec / 3600.0
    half_ra = half_dec / max(np.cos(np.radians(dec)), 1e-3)
    found = _sparcl().find(
        outfields=["sparcl_id", "targetid", "ra", "dec", "redshift", "redshift_err", "spectype"],
        constraints={"data_release": [data_release], "ra": [ra - half_ra, ra + half_ra], "dec": [dec - half_dec, dec + half_dec]},
        limit=limit)
    unique = {}
    for r in found.records:
        if r["spectype"] in spectypes:
            unique.setdefault(r["targetid"], r)
    names = ("sparcl_id", "targetid", "ra", "dec", "redshift", "redshift_err", "spectype")
    table = Table(rows=[[r.get(n) for n in names] for r in unique.values()], names=names) if unique else Table(names=names, dtype=[str, int, float, float, float, float, str])
    if len(table):
        sep = SkyCoord(table["ra"] * u.deg, table["dec"] * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).arcsec
        table["sep_arcsec"] = sep
        table = table[sep <= radius_arcsec]
        table.sort("sep_arcsec")
    else:
        table["sep_arcsec"] = np.array([], dtype=float)
    return table


def desi_spectrum(sparcl_id, data_release="DESI-DR1"):
    """``(wavelength [A], flux [1e-17 erg/cm2/s/A], ivar, model)`` of one DESI spectrum."""
    out = _sparcl().retrieve(uuid_list=[str(sparcl_id)], include=["wavelength", "flux", "ivar", "model"], dataset_list=[data_release])
    if not out.records:
        raise ValueError(f"no DESI spectrum with sparcl_id {sparcl_id}")
    r = out.records[0]
    return (np.asarray(r["wavelength"], float), np.asarray(r["flux"], float), np.asarray(r["ivar"], float), np.asarray(r["model"], float))
