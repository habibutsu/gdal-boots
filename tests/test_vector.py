import os.path
import tempfile
import json
import numpy as np
import pytest

from affine import Affine
from gdal_boots import options, GeoInfo
from gdal_boots.gdal import RasterDataset, VectorDataset


def test_open_file(minsk_boundary_geojson):
    ds = VectorDataset.open(minsk_boundary_geojson)

    assert len(ds.layers) == 1
    assert len(ds.layers[0].features) == 38
    assert ds.layers[0].features[0]["name:en"] == "Minsk"


def test_read_from_bytes(minsk_boundary_gpkg):

    with open(minsk_boundary_gpkg, "rb") as fd:
        data = fd.read()

    ds = VectorDataset.from_bytes(data)
    assert len(ds.layers) == 5


@pytest.mark.skipif(
    not os.path.exists("tests/fixtures/extra/B05_20m.jp2"),
    reason='extra file "tests/fixtures/extra/B05_20m.jp2" does not exist',
)
def test_rasterize():
    vec_ds = VectorDataset.open("tests/fixtures/minsk-polygon.geojson")

    ref_ds = RasterDataset.open("tests/fixtures/extra/B05_20m.jp2")

    with tempfile.NamedTemporaryFile(suffix=".tiff") as fd:
        ds = RasterDataset.create(shape=ref_ds.shape, dtype=ref_ds.dtype, geoinfo=ref_ds.geoinfo)
        vec_ds.rasterize(ds)

        values, counts = np.unique(ds[:], return_counts=True)
        assert np.all(values == [0, 1])
        assert np.all(counts == [29511509, 628591])

        ds.to_file(fd.name, options.GTiff())


def test_rasterize_basic():
    geometry = {
        "type": "Polygon", "coordinates": [ [
            [ 0.6, 1.0 ],
            [ 0.6, 0.75 ],
            [ 0.0, 0.75 ],
            [ 0.0, 1.0 ],
            [ 0.6, 1.0 ]
        ] ]
    }
    geoinfo = GeoInfo(
        epsg=4326,
        transform=Affine(0.01, 0.0, 0.0, 0.0, -0.01, 1.0)
    )
    shape = (100, 100)

    vect_ds = VectorDataset.open(json.dumps(geometry))
    if geoinfo.epsg != 4326:
        vect_ds.ds.GetLayer(0).GetSpatialRef().ImportFromEPSG(geoinfo.epsg)

    mask_ds = RasterDataset.create(shape, np.uint8, geoinfo=geoinfo)
    vect_ds.rasterize(mask_ds, all_touched=False)
    assert mask_ds[:25,:60].all()

    vect_ds.rasterize(mask_ds, all_touched=True)
    assert mask_ds[:26,:61].all()
