import os.path
import tempfile

import pytest

from gdal_boots import options
from gdal_boots.gdal import RasterDataset, VectorDataset


def test_open_file(minsk_boundary_geojson):
    ds = VectorDataset.open(minsk_boundary_geojson)

    assert len(ds.layers) == 1
    assert len(ds.layers[0].features) == 38
    assert ds.layers[0].features[0]['name:en'] == 'Minsk'


@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/B05_20m.jp2'),
    reason='extra file "tests/fixtures/extra/B05_20m.jp2" does not exist',
)
def test_rasterize():
    vec_ds = VectorDataset.open('tests/fixtures/minsk-polygon.geojson')

    ref_ds = RasterDataset.open('tests/fixtures/extra/B05_20m.jp2')

    with tempfile.NamedTemporaryFile(suffix='.png') as fd:
        ds = RasterDataset.create(shape=ref_ds.shape, dtype=ref_ds.dtype, geoinfo=ref_ds.geoinfo)
        vec_ds.rasterize(ds)
        ds.to_file(fd.name, options.GTiff())
