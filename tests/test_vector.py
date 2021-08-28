import tempfile

from build.lib.gdal_boots import options
from build.lib.gdal_boots.gdal import RasterDataset
from gdal_boots.gdal import (
    VectorDataset,
    GeoInfo
)


def test_open_file(minsk_boundary_geojson):

    ds = VectorDataset.open(minsk_boundary_geojson)

    assert len(ds.layers) == 1
    assert len(ds.layers[0].features) == 38
    assert ds.layers[0].features[0]['name:en'] == 'Minsk'


def test_rasterize():
    vec_ds = VectorDataset.open('tests/fixtures/minsk-polygon.geojson')

    ref_ds = RasterDataset.open('tests/fixtures/extra/B05_20m.jp2')

    with tempfile.NamedTemporaryFile(suffix='.png') as fd:

        ds = RasterDataset.create(shape=ref_ds.shape, dtype=ref_ds.dtype, geoinfo=ref_ds.geoinfo)
        vec_ds.rasterize(ds)
        ds.to_file(fd.name, options.GTiff())