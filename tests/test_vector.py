import json
import os.path
import tempfile

import numpy as np
import pytest
from affine import Affine

from gdal_boots import GeoInfo, gdal_version, options
from gdal_boots.gdal import RasterDataset, VectorDataset


def test_open_file(minsk_boundary_geojson):
    ds = VectorDataset.open(minsk_boundary_geojson)

    assert len(ds.layers) == 1
    assert len(ds.layers[0].features) == 38
    assert ds.layers[0].features[0]["name:en"] == "Minsk"


def test_to_epsg_gpkg(minsk_boundary_gpkg):
    vds = VectorDataset.open(minsk_boundary_gpkg)
    vds_3857 = vds.to_epsg(3857)
    with tempfile.NamedTemporaryFile(suffix=".gpkg") as fd:
        vds_3857.to_file(fd.name, options.GPKG())


def test_to_epsg_geojson(minsk_boundary_geojson):
    vds = VectorDataset.open(minsk_boundary_geojson)
    vds_3857 = vds.to_epsg(3857)
    with tempfile.NamedTemporaryFile(suffix=".geojson") as fd:
        vds_3857.to_file(fd.name, options.GPKG())


def test_read_from_bytes(minsk_boundary_gpkg):
    with open(minsk_boundary_gpkg, "rb") as fd:
        data = fd.read()

    vds = VectorDataset.from_bytes(data)
    assert len(vds.layers) == 5

    layer = vds.layers.first()

    count = 0
    for feature in layer.features:
        assert feature.keys() == [
            "osm_id",
            "name",
            "barrier",
            "highway",
            "ref",
            "address",
            "is_in",
            "place",
            "man_made",
            "other_tags",
        ]
        count += 1

    assert count == 7


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
    geometry = {"type": "Polygon", "coordinates": [[[0.6, 1.0], [0.6, 0.75], [0.0, 0.75], [0.0, 1.0], [0.6, 1.0]]]}
    geoinfo = GeoInfo(epsg=4326, transform=Affine(0.01, 0.0, 0.0, 0.0, -0.01, 1.0))
    shape = (100, 100)

    vect_ds = VectorDataset.open(json.dumps(geometry), srs=geoinfo.srs)

    mask_ds = RasterDataset.create(shape, np.uint8, geoinfo=geoinfo)
    vect_ds.rasterize(mask_ds, all_touched=False)
    assert mask_ds[:25, :60].all()

    if gdal_version < (3, 6, 3):
        pytest.skip("known bug connected with all_touched=True")

    vect_ds.rasterize(mask_ds, all_touched=True)
    assert mask_ds[:25, :60].all()


def test_fields():

    vds = VectorDataset.create()

    layer = vds.add_layer("test-layer", VectorDataset.GeometryType.Point, 3857)
    layer.add_field("string_property", str)
    layer.add_field("int_property", int)

    assert layer.field_names == ["string_property", "int_property"]
    assert layer.field_types == [str, int]
