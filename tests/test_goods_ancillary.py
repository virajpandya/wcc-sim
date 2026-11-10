"""Offline tests of scripts/goods_ancillary.py: 3D-HST catalog assembly, positional matching and the DESI (SPARCL) queries, all faked."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import goods_ancillary as ga  # noqa: E402


def _fits_bytes(table):
    buf = io.BytesIO()
    table.write(buf, format="fits")
    return buf.getvalue()


def test_threedhst_urls():
    base = "https://archive.stsci.edu/missions/hlsp/3d-hst/RELEASE_V4.0/Photometry/"
    assert ga.threedhst_url("goods-n", "fout") == base + "GOODS-N/goodsn_3dhst.v4.1.cats/Fast/goodsn_3dhst.v4.1.fout.FITS"
    assert ga.threedhst_url("goods-s", "zout").endswith("GOODS-S/goodss_3dhst.v4.1.cats/Eazy/goodss_3dhst.v4.1.zout.FITS")
    assert ga.threedhst_url("goods-n", "cat").endswith("/Catalog/goodsn_3dhst.v4.1.cat.FITS")
    with pytest.raises(ValueError, match="field"):
        ga.threedhst_url("cosmos", "cat")
    with pytest.raises(ValueError, match="part"):
        ga.threedhst_url("goods-n", "pz")


def test_fetch_3dhst_joins_the_three_catalogs_and_caches(tmp_path, monkeypatch):
    ids = np.array([3, 1, 2])
    parts = {
        "cat": Table({"id": ids, "ra": [10.0, 10.1, 10.2], "dec": [5.0, 5.1, 5.2], "z_spec": [-1.0, 0.5, 1.0], "star_flag": [0, 0, 1], "extra": [1, 2, 3]}),
        "zout": Table({"id": ids, "z_peak": [0.9, 0.52, 1.1]}),
        "fout": Table({"id": ids, "lmass": [9.0, 10.0, 11.0], "lsfr": [0.1, -99.0, 1.0]}),
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        part = url.rsplit(".", 2)[-2]
        return SimpleNamespace(content=_fits_bytes(parts[part]))

    monkeypatch.setattr(ga, "_get", fake_get)
    out = ga.fetch_3dhst("goods-n", cache_dir=str(tmp_path))
    assert out.colnames == ["id", "ra", "dec", "z_spec", "star_flag", "z_peak", "lmass", "lsfr"]
    assert list(out["id"]) == [1, 2, 3]
    assert out["lmass"][0] == 10.0 and out["z_peak"][2] == 0.9
    assert len(calls) == 3
    again = ga.fetch_3dhst("goods-n", cache_dir=str(tmp_path))          # from the cache: no new download
    assert len(calls) == 3 and len(again) == 3


def test_match_catalog_limits_the_separation():
    table = Table({"ra": [10.0, 10.0, 11.0], "dec": [5.0, 5.001, 5.0]})
    idx, sep = ga.match_catalog([10.0, 10.00005, 12.0], [5.0004, 5.0, 5.0], table, max_sep_arcsec=1.5)
    assert list(idx) == [0, 0, -1]
    assert sep[0] == pytest.approx(1.44, abs=0.02) and sep[2] > 1000


class _FakeSparcl:
    def __init__(self, records):
        self.records, self.find_args = records, None

    def find(self, outfields, constraints, limit):
        self.find_args = constraints
        return SimpleNamespace(records=self.records)

    def retrieve(self, uuid_list, include, dataset_list):
        assert uuid_list == ["abc"] and dataset_list == ["DESI-DR1"]
        return SimpleNamespace(records=[{"wavelength": [4000.0, 4001.0], "flux": [1.0, 2.0], "ivar": [1.0, 1.0], "model": [1.5, 1.5]}])


def _record(targetid, ra, dec, spectype="GALAXY", z=0.5):
    return {"sparcl_id": f"id{targetid}", "targetid": targetid, "ra": ra, "dec": dec, "redshift": z, "redshift_err": 1e-4, "spectype": spectype}


def test_desi_galaxy_spectra_filters_dedupes_and_sorts(monkeypatch):
    fake = _FakeSparcl([_record(1, 10.001, 5.0), _record(1, 10.001, 5.0), _record(2, 10.0002, 5.0), _record(3, 10.0, 5.0, "STAR"),
                        _record(4, 10.05, 5.0)])
    monkeypatch.setattr(ga, "_sparcl", lambda: fake)
    out = ga.desi_galaxy_spectra(10.0, 5.0, 10.0)
    assert list(out["targetid"]) == [2, 1]                       # nearest first; the duplicate, the star and the target 180" away are gone
    assert out["sep_arcsec"][0] < out["sep_arcsec"][1] <= 10.0
    assert fake.find_args["data_release"] == ["DESI-DR1"]


def test_desi_galaxy_spectra_empty(monkeypatch):
    monkeypatch.setattr(ga, "_sparcl", lambda: _FakeSparcl([]))
    out = ga.desi_galaxy_spectra(10.0, 5.0, 10.0)
    assert len(out) == 0 and "sep_arcsec" in out.colnames


def test_desi_spectrum_returns_arrays(monkeypatch):
    monkeypatch.setattr(ga, "_sparcl", lambda: _FakeSparcl([]))
    wave, flux, ivar, model = ga.desi_spectrum("abc")
    assert wave.tolist() == [4000.0, 4001.0] and flux.tolist() == [1.0, 2.0] and model.tolist() == [1.5, 1.5]


def test_missing_sparcl_client_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sparcl", None)
    monkeypatch.setitem(sys.modules, "sparcl.client", None)
    with pytest.raises(ImportError, match="sparclclient"):
        ga._sparcl()
