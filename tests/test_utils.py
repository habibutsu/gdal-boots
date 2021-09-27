import os.path
import tempfile

import numpy as np
import pytest
import shapely.geometry
from affine import Affine

from gdal_boots import PNG, GeoInfo, GTiff, RasterDataset


@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/B04.tif'),
    reason='extra file "tests/fixtures/extra/B04.tif" does not exist',
)
def test_read_by_geom(minsk_polygon):
    bbox = shapely.geometry.shape(minsk_polygon).bounds

    with tempfile.TemporaryDirectory() as tmp_dir:
        with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
            rgba_ds = RasterDataset.create((4, *ds.shape), ds.dtype, ds.geoinfo)
            rgba_ds[0, :] = ds[:]
            rgba_ds[1, :] = ds[:]
            rgba_ds[2, :] = ds[:]
            # no transparency
            rgba_ds[3, :] = np.iinfo(rgba_ds.dtype).max

            cropped_ds, mask = rgba_ds.crop_by_geometry(minsk_polygon)
            cropped_ds.to_file(f'{tmp_dir}/cropped_by_polygon.png', PNG())
            cropped_ds.to_file(f'{tmp_dir}/warped_by_mask.tif', GTiff())
