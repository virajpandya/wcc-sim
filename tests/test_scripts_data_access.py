"""Offline tests of the data-access helpers in scripts/ (no network, faked responses).

* legacysurvey_galaxy_mock.fetch_tractor: Data Lab first, retries, fall back to the
  legacysurvey.org viewer, caching, column/row handling of the viewer table.
* hst_cutouts (CANDELS, UVCANDELS and GOODS): URL layout of the five CANDELS fields, validation, the WCS of
  headers that carry distortion keywords, and byte-range cutouts from a synthetic mosaic.
"""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hst_cutouts as hc  # noqa: E402
import legacysurvey_galaxy_mock as lsm  # noqa: E402

COLS = lsm.TRACTOR_COLUMNS


def _tractor_table(n=3, source_tag=0.0):
    return Table({
        "ra": np.full(n, 150.1), "dec": np.full(n, 2.2),
        "type": np.array(["EXP", "DEV", "PSF", "REX"][:n]),
        "shape_r": np.linspace(0.5, 1.5, n), "shape_e1": np.zeros(n), "shape_e2": np.zeros(n),
        "sersic": np.ones(n), "flux_g": np.full(n, 1.0 + source_tag), "flux_r": np.full(n, 2.0),
        "flux_z": np.full(n, 3.0), "mw_transmission_r": np.full(n, 0.96),
    })


@pytest.fixture
def fetchers(monkeypatch):
    """Replace both Tractor services; the returned dict counts calls and sets behaviour."""
    state = {"datalab": 0, "viewer": 0, "datalab_fails": False, "viewer_fails": False}

    def datalab(release, ra, dec, radius, timeout):
        state["datalab"] += 1
        if state["datalab_fails"]:
            raise OSError("502 Bad Gateway")
        return _tractor_table(source_tag=0.0)

    def viewer(release, ra, dec, radius, timeout):
        state["viewer"] += 1
        if state["viewer_fails"]:
            raise OSError("viewer down")
        return _tractor_table(source_tag=1.0)

    monkeypatch.setattr(lsm, "_tractor_datalab", datalab)
    monkeypatch.setattr(lsm, "_tractor_viewer", viewer)
    return state


class TestFetchTractor:
    def test_datalab_answers_and_viewer_is_not_called(self, fetchers):
        t = lsm.fetch_tractor(150.1, 2.2, 30.0, backoff_s=0)
        assert fetchers == {**fetchers, "datalab": 1, "viewer": 0}
        assert t.meta["source"] == "datalab"

    def test_falls_back_to_viewer_after_retries_with_a_warning(self, fetchers):
        fetchers["datalab_fails"] = True
        with pytest.warns(UserWarning, match="came from viewer"):
            t = lsm.fetch_tractor(150.1, 2.2, 30.0, retries=2, backoff_s=0)
        assert fetchers["datalab"] == 3 and fetchers["viewer"] == 1      # 1 try + 2 retries, then the viewer
        assert t.meta["source"] == "viewer"
        assert float(t["flux_g"][0]) == 2.0                                # the viewer's table

    def test_both_failing_raises_with_every_error(self, fetchers):
        fetchers["datalab_fails"] = fetchers["viewer_fails"] = True
        with pytest.raises(RuntimeError) as err:
            lsm.fetch_tractor(150.1, 2.2, 30.0, retries=0, backoff_s=0)
        assert "datalab" in str(err.value) and "viewer" in str(err.value)

    def test_single_source_does_not_fall_back(self, fetchers):
        fetchers["datalab_fails"] = True
        with pytest.raises(RuntimeError):
            lsm.fetch_tractor(150.1, 2.2, 30.0, source="datalab", retries=0, backoff_s=0)
        assert fetchers["viewer"] == 0
        t = lsm.fetch_tractor(150.1, 2.2, 30.0, source="viewer", backoff_s=0)
        assert t.meta["source"] == "viewer"

    def test_unknown_source_is_rejected(self, fetchers):
        with pytest.raises(ValueError, match="source"):
            lsm.fetch_tractor(150.1, 2.2, 30.0, source="mars")

    def test_result_is_cached_and_the_cache_needs_no_network(self, fetchers, tmp_path):
        first = lsm.fetch_tractor(150.1, 2.2, 30.0, cache_dir=str(tmp_path), backoff_s=0)
        fetchers["datalab_fails"] = fetchers["viewer_fails"] = True
        again = lsm.fetch_tractor(150.1, 2.2, 30.0, cache_dir=str(tmp_path), backoff_s=0)
        assert len(again) == len(first) and fetchers["datalab"] == 1


class _Resp:
    def __init__(self, content=b"", status=200):
        self.content, self.status_code = content, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


class TestViewerTable:
    def _viewer_fits(self):
        tab = _tractor_table(4)
        tab["type"] = np.array(["EXP ", "DEV ", "PSF ", "REX "])      # padded, as in the FITS table
        tab["brick_primary"] = np.array([True, False, True, True])
        tab["objid"] = np.arange(4)                                     # extra columns are dropped
        buf = io.BytesIO()
        tab.write(buf, format="fits")
        return buf.getvalue()

    def test_keeps_primary_rows_strips_types_and_selects_columns(self, monkeypatch):
        calls = []
        monkeypatch.setattr(lsm.requests, "get", lambda url, params=None, timeout=None: calls.append((url, params)) or _Resp(self._viewer_fits()))
        t = lsm._tractor_viewer("ls_dr10", 150.1, 2.2, 30.0, timeout=5)
        assert t.colnames == COLS
        assert list(t["type"]) == ["EXP", "PSF", "REX"]                 # the brick_primary == False row is gone
        assert "ls-dr10" in calls[0][0] and len(calls) == 1

    def test_box_that_wraps_past_ra_0_is_two_requests(self, monkeypatch):
        calls = []
        monkeypatch.setattr(lsm.requests, "get", lambda url, params=None, timeout=None: calls.append(params) or _Resp(self._viewer_fits()))
        t = lsm._tractor_viewer("ls_dr10", 0.001, 2.2, 30.0, timeout=5)
        assert len(calls) == 2 and len(t) == 6
        assert float(calls[0]["rahi"]) == 360.0 and float(calls[1]["ralo"]) == 0.0


# --------------------------------------------------------------------------- hst_cutouts: CANDELS and UVCANDELS


class TestMosaicUrl:
    ROOT = "https://archive.stsci.edu/pub/hlsp/candels/"

    def test_flat_layout_fields(self):
        assert hc.mosaic_url("cosmos", "f606w") == (
            self.ROOT + "cosmos/cos-tot/v1.0/hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits")
        assert hc.mosaic_url("goods-s", "f105w", "exp") == (
            self.ROOT + "goods-s/gs-tot/v1.0/hlsp_candels_hst_wfc3_gs-tot_f105w_v1.0_exp.fits")
        assert hc.mosaic_url("uds", "f160w").endswith("uds/uds-tot/v1.0/hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits")

    def test_split_layout_fields(self):
        assert hc.mosaic_url("egs", "f606w") == (
            self.ROOT + "egs/egs-tot/v1.0/30mas-acs/hlsp_candels_hst_acs_egs-tot-30mas_f606w_v1.0_drz.fits")
        assert hc.mosaic_url("egs", "f606w", acs_scale="60mas") == (
            self.ROOT + "egs/egs-tot/v1.0/60mas-all/hlsp_candels_hst_acs_egs-tot-60mas_f606w_v1.0_drz.fits")
        assert hc.mosaic_url("goods-n", "f160w") == (
            self.ROOT + "goods-n/gn-tot/v1.0/60mas-all/hlsp_candels_hst_wfc3_gn-tot-60mas_f160w_v1.0_drz.fits")

    @pytest.mark.parametrize("args", [("mars", "f606w"), ("egs", "f850lp"), ("cosmos", "f105w"), ("cosmos", "r")])
    def test_unknown_field_or_band(self, args):
        with pytest.raises(ValueError):
            hc.mosaic_url(*args)

    def test_unknown_kind_and_scale(self):
        with pytest.raises(ValueError, match="kind"):
            hc.mosaic_url("cosmos", "f606w", kind="foo")
        with pytest.raises(ValueError, match="acs_scale"):
            hc.mosaic_url("egs", "f606w", acs_scale="10mas")

    def test_every_field_lists_only_buildable_bands(self):
        for field, bands in hc.list_fields().items():
            for band in bands:
                root = hc.UVCANDELS_ROOT + field if band == "f435w" else self.ROOT + FIELD_DIR[field]   # F435W is UVCANDELS
                assert hc.mosaic_url(field, band).startswith(root)


FIELD_DIR = {"cosmos": "cosmos/cos-tot", "uds": "uds/uds-tot", "egs": "egs/egs-tot",
             "goods-n": "goods-n/gn-tot", "goods-s": "goods-s/gs-tot"}


def _mosaic_header(nx, ny, scale_arcsec=0.03, extra=None, rotation_deg=0.0):
    hdr = fits.Header()
    phi = np.radians(rotation_deg)
    hdr["SIMPLE"], hdr["BITPIX"], hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = True, -32, 2, nx, ny
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRVAL1"], hdr["CRVAL2"], hdr["CRPIX1"], hdr["CRPIX2"] = 150.0, 2.0, nx / 2 + 0.5, ny / 2 + 0.5
    s = scale_arcsec / 3600
    hdr["CD1_1"], hdr["CD1_2"] = -s * np.cos(phi), s * np.sin(phi)       # +y axis at PA = rotation_deg, east of north
    hdr["CD2_1"], hdr["CD2_2"] = s * np.sin(phi), s * np.cos(phi)
    hdr["PHOTFLAM"], hdr["PHOTPLAM"] = 7.0e-20, 6000.0
    for k, v in (extra or {}).items():
        hdr[k] = v
    return hdr


class TestCelestialWcs:
    def test_distortion_keywords_are_ignored(self):
        extra = {"CPDIS1": "Lookup", "DP1": "EXTVER: 1", "D2IMDIS1": "Lookup", "A_ORDER": 2, "A_0_2": 1e-3,
                 "B_ORDER": 2, "AP_ORDER": 2, "BP_ORDER": 2}
        wcs = hc._celestial_wcs(_mosaic_header(100, 80, extra=extra))
        assert wcs.sip is None
        ra, dec = wcs.pixel_to_world_values(49.5, 39.5)         # CRPIX is 1-based: (50.5, 40.5)
        assert ra == pytest.approx(150.0, abs=1e-6) and dec == pytest.approx(2.0, abs=1e-6)


class _Server:
    """Serves one synthetic mosaic over fake HTTP with byte ranges, counting requests."""

    def __init__(self, nx=200, ny=120, header_extra=None, rotation_deg=0.0):
        yy, xx = np.indices((ny, nx))
        self.data = (xx + 1000 * yy).astype(">f4")                       # every pixel value tells its position
        hdr = _mosaic_header(nx, ny, extra=header_extra, rotation_deg=rotation_deg)
        buf = io.BytesIO()
        fits.PrimaryHDU(self.data.astype(np.float32), header=hdr).writeto(buf)
        self.blob, self.nx, self.ny, self.requests = buf.getvalue(), nx, ny, []

    def get(self, url, headers=None, timeout=None, **kw):
        rng = (headers or {}).get("Range", "")
        lo, hi = (int(v) for v in rng.removeprefix("bytes=").split("-"))
        self.requests.append((lo, hi))
        return _Resp(self.blob[lo:hi + 1], 206)


@pytest.fixture
def server(monkeypatch):
    srv = _Server(header_extra={"CPDIS1": "Lookup", "A_ORDER": 2, "A_0_2": 1e-3})
    monkeypatch.setattr(hc.requests, "get", srv.get)
    return srv


class TestFetchCutout:
    def test_cutout_has_the_right_pixels_wcs_and_zero_point(self, server):
        # a 3" x 1.8" patch = 100 x 60 px of 0.03" at the mosaic centre
        data, wcs, zp, hdr = hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w")
        assert data.shape == (60, 100)
        x0, y0 = int(data[0, 0] % 1000), int(data[0, 0] // 1000)       # pixel values encode their position
        assert abs(x0 - 50) <= 1 and abs(y0 - 30) <= 1                   # centred on the target, to a pixel
        assert data[-1, -1] == pytest.approx(x0 + 99 + 1000 * (y0 + 59))
        assert zp == pytest.approx(-2.5 * np.log10(7.0e-20) - 5 * np.log10(6000.0) - 2.408)
        cx, cy = wcs.world_to_pixel_values(150.0, 2.0)
        assert cx == pytest.approx(49.5, abs=1.0) and cy == pytest.approx(29.5, abs=1.0)

    def test_only_the_needed_rows_are_requested(self, server):
        hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w")
        header_read, rows_read = server.requests[0], server.requests[-1]
        assert header_read[0] == 0
        assert (rows_read[1] - rows_read[0] + 1) == 60 * server.nx * 4        # 60 rows, full width

    def test_cache_round_trip_needs_no_network(self, server, tmp_path, monkeypatch):
        d1, _, zp1, _ = hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", cache_dir=str(tmp_path))
        monkeypatch.setattr(hc.requests, "get", lambda *a, **k: pytest.fail("network used"))
        d2, _, zp2, _ = hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", cache_dir=str(tmp_path))
        np.testing.assert_array_equal(d1, d2)
        assert zp1 == zp2

    def test_cosmos_cache_names_are_unchanged(self, server, tmp_path):
        hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", cache_dir=str(tmp_path))
        hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", cache_dir=str(tmp_path), kind="exp")
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["candels_cos_f606w_150.00000_+2.00000_3.0x1.8.fits",
                         "candels_cos_f606w_exp_150.00000_+2.00000_3.0x1.8.fits"]

    def test_exposure_maps_have_no_zero_point(self, server):
        data, _, zp, hdr = hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", kind="exp")
        assert hdr["BUNIT"] == "s"            # the synthetic header still has PHOTFLAM, so zp is finite here
        assert data.shape == (60, 100)

    def test_outside_the_mosaic_raises(self, server):
        with pytest.raises(ValueError, match="outside"):
            hc.fetch_candels_cutout(150.0 + 10 / 3600, 2.0, 3.0, 1.8, "f606w")
        with pytest.raises(ValueError, match="far from"):
            hc.fetch_candels_cutout(10.0, -80.0, 3.0, 1.8, "f606w")

    def test_download_cap(self, server):
        with pytest.raises(ValueError, match="max_download_mb"):
            hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", max_download_mb=0.001)

    def test_mostly_empty_patch_warns(self, monkeypatch):
        srv = _Server()
        srv.data[:] = 0.0
        hdr = _mosaic_header(srv.nx, srv.ny)
        buf = io.BytesIO()
        fits.PrimaryHDU(srv.data.astype(np.float32), header=hdr).writeto(buf)
        srv.blob = buf.getvalue()
        monkeypatch.setattr(hc.requests, "get", srv.get)
        with pytest.warns(UserWarning, match="empty"):
            hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w")

    def test_missing_mosaic_is_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(hc.requests, "get", lambda *a, **k: _Resp(b"", 404))
        with pytest.raises(ValueError, match="no such mosaic"):
            hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w", kind="exp")

    def test_mosaic_info(self, server):
        info = hc.mosaic_info("cosmos", "f606w")
        assert (info["nx"], info["ny"]) == (server.nx, server.ny)
        assert info["pixel_scale_arcsec"] == pytest.approx(0.03, rel=1e-6)
        assert info["centre_ra"] == pytest.approx(150.0, abs=1e-3) and info["centre_dec"] == pytest.approx(2.0, abs=1e-3)


class TestGetRetries:
    """hc.get_with_retry: retry connection errors, timeouts and 5xx; never a 404."""

    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(hc.time, "sleep", lambda s: None)

    def _flaky(self, monkeypatch, outcomes):
        calls = []

        def get(url, headers=None, timeout=None, **kw):
            calls.append(url)
            out = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
            if isinstance(out, Exception):
                raise out
            return _Resp(b"ok", out)

        monkeypatch.setattr(hc.requests, "get", get)
        return calls

    def test_recovers_from_connection_errors(self, monkeypatch):
        calls = self._flaky(monkeypatch, [hc.requests.ConnectionError("refused"), hc.requests.Timeout("slow"), 200])
        assert hc.get_with_retry("u").content == b"ok" and len(calls) == 3

    def test_retries_server_errors_but_not_a_missing_file(self, monkeypatch):
        calls = self._flaky(monkeypatch, [503, 200])
        assert hc.get_with_retry("u").status_code == 200 and len(calls) == 2
        calls = self._flaky(monkeypatch, [404])
        assert hc.get_with_retry("u").status_code == 404 and len(calls) == 1

    def test_gives_up_after_the_attempts(self, monkeypatch):
        calls = self._flaky(monkeypatch, [hc.requests.ConnectionError("down")])
        with pytest.raises(hc.requests.ConnectionError):
            hc.get_with_retry("u", attempts=3)
        assert len(calls) == 3
        calls = self._flaky(monkeypatch, [502])
        assert hc.get_with_retry("u", attempts=2).status_code == 502 and len(calls) == 2   # the caller sees the last answer

    def test_a_flaky_archive_does_not_stop_a_cutout(self, server, monkeypatch):
        real, state = hc.requests.get, {"n": 0}

        def flaky(*a, **k):
            state["n"] += 1
            if state["n"] in (1, 3):
                raise hc.requests.ConnectionError("refused")
            return real(*a, **k)

        monkeypatch.setattr(hc.requests, "get", flaky)
        monkeypatch.setattr(hc.time, "sleep", lambda s: None)
        data, *_ = hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w")
        assert data.shape == (60, 100) and state["n"] >= 4


class TestRotation:
    def test_mosaic_info_reports_the_rotation(self, monkeypatch):
        for phi in (0.0, 30.0, -49.7):
            srv = _Server(rotation_deg=phi)
            monkeypatch.setattr(hc.requests, "get", srv.get)
            assert hc.mosaic_info("egs", "f606w")["rotation_deg"] == pytest.approx(phi, abs=1e-6)

    def test_bounding_cutout_size(self):
        assert hc.bounding_cutout_size(50.0, 30.0, 0.0) == pytest.approx((50.0, 30.0))
        assert hc.bounding_cutout_size(50.0, 30.0, 90.0) == pytest.approx((30.0, 50.0))
        assert hc.bounding_cutout_size(50.0, 30.0, -90.0) == pytest.approx((30.0, 50.0))
        w, h = hc.bounding_cutout_size(50.0, 30.0, 45.0)
        assert w == pytest.approx(h) and w == pytest.approx(80.0 / np.sqrt(2.0))

    def test_the_bounding_cutout_really_contains_a_rotated_frame(self):
        # corners of a north-up 50x30 rectangle, expressed in the axes of a mosaic rotated by phi
        w, h = 50.0, 30.0
        for phi in (0.0, 20.0, -49.7, 90.0):
            c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
            corners = np.array([(sx * w / 2, sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)])
            x = corners[:, 0] * c + corners[:, 1] * s
            y = -corners[:, 0] * s + corners[:, 1] * c
            bw, bh = hc.bounding_cutout_size(w, h, phi)
            assert abs(x).max() == pytest.approx(bw / 2) and abs(y).max() == pytest.approx(bh / 2)


# --------------------------------------------------------------------------- hst_cutouts: GOODS


class TestGoodsUrls:
    def test_tile_urls(self):
        assert hc.goods_tile_url("goods-s", "f435w", "sect23") == hc.GOODS_ROOT + "h_sb_sect23_v2.0_drz_img.fits"
        assert hc.goods_tile_url("goods-n", "f775w", "sect11", "wht") == hc.GOODS_ROOT + "h_ni_sect11_v2.0_wht_img.fits"
        assert hc.goods_tile_url("GOODS-S", "F850LP", "sect45").endswith("h_sz_sect45_v2.0_drz_img.fits")

    @pytest.mark.parametrize("args", [("cosmos", "f606w"), ("goods-s", "f814w"), ("goods-s", "f606w", "exp")])
    def test_bad_field_band_or_kind(self, args):
        with pytest.raises(ValueError):
            hc.goods_tile_url(args[0], args[1], "sect11", *args[2:])

    def test_the_documented_calibration_is_complete(self):
        assert set(hc.GOODS_ZP_AB) == set(hc.GOODS_BANDS)
        for field in hc.GOODS_FIELDS:
            assert set(hc.GOODS_TYPICAL_EXPOSURE_S[field]) == set(hc.GOODS_BANDS)

    def test_list_sections_reads_the_archive_listing(self, monkeypatch):
        hc.list_goods_sections.cache_clear()
        page = ('<a href="h_sb_sect11_v2.0_drz_img.fits">x</a><a href="h_sb_sect12_v2.0_drz_img.fits">x</a>'
                '<a href="h_sb_sect12_v2.0_wht_img.fits">x</a><a href="h_nb_sect13_v2.0_drz_img.fits">x</a>')
        monkeypatch.setattr(hc, "get_with_retry", lambda url, **k: _Resp(page.encode(), 200))
        monkeypatch.setattr(_Resp, "text", property(lambda self: self.content.decode()), raising=False)
        assert hc.list_goods_sections("goods-s", "f435w") == ("sect11", "sect12")
        hc.list_goods_sections.cache_clear()


class TestGoodsTiles:
    """Two contiguous 8192-pixel tiles stacked in y on one pixel grid, as in the GOODS release."""

    @pytest.fixture(autouse=True)
    def _tiles(self, monkeypatch):
        hc.list_goods_sections.cache_clear()
        hc._goods_tile_geometry.cache_clear()
        from astropy.wcs import WCS

        def geometry(field, band, section):
            w = WCS(naxis=2)
            w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            w.wcs.crval = [53.0, -27.8]
            w.wcs.cd = [[-0.03 / 3600, 0], [0, 0.03 / 3600]]
            w.wcs.crpix = [4096.5, 4096.5 - (8192 if section == "sect12" else 0)]      # sect12 lies one tile north of sect11
            return 8192, 8192, w

        monkeypatch.setattr(hc, "list_goods_sections", lambda field, band: ("sect11", "sect12"))
        monkeypatch.setattr(hc, "_goods_tile_geometry", geometry)

    def test_the_tile_containing_a_patch(self):
        assert hc.find_goods_tile("goods-s", "f606w", 53.0, -27.8, 180.0, 180.0) == "sect11"        # centre of sect11
        dec_north = -27.8 + 8192 * 0.03 / 3600                                                 # centre of sect12
        assert hc.find_goods_tile("goods-s", "f606w", 53.0, dec_north, 180.0, 180.0) == "sect12"

    def test_a_patch_across_a_tile_boundary_is_refused(self):
        dec_edge = -27.8 + 4096 * 0.03 / 3600                                                  # the boundary between the tiles
        with pytest.raises(ValueError, match="crosses the boundary"):
            hc.find_goods_tile("goods-s", "f606w", 53.0, dec_edge, 180.0, 180.0)

    def test_a_patch_outside_the_mosaic_is_refused(self):
        with pytest.raises(ValueError, match="outside"):
            hc.find_goods_tile("goods-s", "f606w", 10.0, 40.0, 180.0, 180.0)


class TestGoodsCutout:
    @pytest.fixture
    def tile_server(self, monkeypatch):
        hc.list_goods_sections.cache_clear()
        hc._goods_tile_geometry.cache_clear()
        srv = _Server(nx=200, ny=120)
        monkeypatch.setattr(hc.requests, "get", srv.get)
        monkeypatch.setattr(hc, "list_goods_sections", lambda field, band: ("sect11",))
        return srv

    def test_cutout_values_zero_point_and_cache_name(self, tile_server, tmp_path):
        data, wcs, zp, hdr = hc.fetch_goods_cutout(150.0, 2.0, 3.0, 1.8, "f435w", field="goods-s", cache_dir=str(tmp_path))
        assert data.shape == (60, 100) and zp == hc.GOODS_ZP_AB["f435w"] == pytest.approx(25.673)
        assert hdr["BAND"] == "F435W" and hdr["BUNIT"] == "ELECTRONS/S"
        assert [p.name for p in tmp_path.iterdir()] == ["goods_s_f435w_150.00000_+2.00000_3.0x1.8.fits"]
        again = hc.fetch_goods_cutout(150.0, 2.0, 3.0, 1.8, "f435w", field="goods-s", cache_dir=str(tmp_path))
        np.testing.assert_array_equal(again[0], data)

    def test_weight_maps_have_no_zero_point(self, tile_server):
        _, _, zp, hdr = hc.fetch_goods_cutout(150.0, 2.0, 3.0, 1.8, "f606w", field="goods-n", kind="wht")
        assert np.isnan(zp) and hdr["FIELD"] == "goods-n"


class TestCutoutFromUrl:
    def test_the_shared_core_returns_pixels_and_a_cutout_wcs(self, monkeypatch):
        srv = _Server()
        monkeypatch.setattr(hc.requests, "get", srv.get)
        data, wcs, header = hc.cutout_from_url("http://x/y.fits", 150.0, 2.0, 3.0, 1.8, label="a test mosaic")
        assert data.shape == (60, 100) and data.dtype == np.float32 and header["NAXIS1"] == srv.nx
        cx, cy = wcs.world_to_pixel_values(150.0, 2.0)
        assert cx == pytest.approx(49.5, abs=1.0) and cy == pytest.approx(29.5, abs=1.0)
        with pytest.raises(ValueError, match="a test mosaic"):
            hc.cutout_from_url("http://x/y.fits", 150.0 + 10 / 3600, 2.0, 3.0, 1.8, label="a test mosaic")


def test_no_unexpected_warnings_on_a_normal_cutout(server):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hc.fetch_candels_cutout(150.0, 2.0, 3.0, 1.8, "f606w")


class TestUvcandelsF435w:
    def test_urls(self):
        base = "https://archive.stsci.edu/hlsps/uvcandels/"
        assert hc.mosaic_url("cosmos", "f435w") == base + "cosmos/hlsp_uvcandels_hst_acs-wfc_cosmos-30mas_f435w_v1.0_sci.fits"
        assert hc.mosaic_url("egs", "f435w", "wht", "60mas") == base + "egs/hlsp_uvcandels_hst_acs-wfc_egs-60mas_f435w_v1.0_wht.fits"
        assert hc.mosaic_url("cosmos", "f435w", "rms").endswith("_f435w_v1.0_rms.fits")

    def test_no_exposure_map_and_only_two_fields(self):
        with pytest.raises(ValueError, match="no exposure map"):
            hc.mosaic_url("cosmos", "f435w", "exp")
        with pytest.raises(ValueError, match="no f435w"):
            hc.mosaic_url("uds", "f435w")
        assert "f435w" in hc.list_fields()["cosmos"] and "f435w" in hc.list_fields()["egs"]

    def test_candels_urls_unchanged(self):
        assert hc.mosaic_url("cosmos", "f814w").endswith("cos-tot_f814w_v1.0_drz.fits")


class TestMaskToSources:
    def test_keeps_the_blob_and_zeroes_the_noise(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (200, 200))
        yy, xx = np.mgrid[:200, :200]
        data += 20.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 6.0**2))
        out = hc.mask_to_sources(data)
        assert out[100, 100] == data[100, 100]                       # the source is kept as it is, noise included
        assert np.all(out[:30] == 0) and np.all(out[:, :30] == 0)    # far from it everything is zero
        assert out.shape == data.shape

    def test_pure_noise_becomes_empty(self):
        out = hc.mask_to_sources(np.random.default_rng(1).normal(0.0, 1.0, (150, 150)))
        assert np.count_nonzero(out) == 0
