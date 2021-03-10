import tempfile
import shapely.geometry
import numpy as np
from affine import Affine

from gdal_boots import (
    RasterDataset,
    GeoInfo,
    PNG,
    GTiff
)


def test_read_by_geom(minsk_polygon):
    bbox = shapely.geometry.shape(minsk_polygon).bounds

    with tempfile.TemporaryDirectory() as tmp_dir:
        with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
            rgba_ds = RasterDataset.create((4, *ds.shape), ds.dtype, ds.geoinfo)
            rgba_ds[0,:] = ds[:]
            rgba_ds[1,:] = ds[:]
            rgba_ds[2,:] = ds[:]
            # no transparency
            rgba_ds[3,:] = np.iinfo(rgba_ds.dtype).max

            cropped_ds, mask = rgba_ds.crop_by_geometry(minsk_polygon)
            cropped_ds.to_file(f'{tmp_dir}/cropped_by_polygon.png', PNG())
            cropped_ds.to_file(f'{tmp_dir}/warped_by_mask.tif', GTiff())


def test_read_by_geom_extra():

    ds1 = RasterDataset.create(
        shape=(1134, 1134),
        dtype=np.uint8,
        geoinfo=GeoInfo(
            epsg=32720,
            transform=Affine(
                10.000000005946216, 0.0, 554680.0000046358,
                0.0, -10.000000003180787, 6234399.99998708
            )
        )
    )
    ds1[:] = np.random.randint(64, 128, (1134, 1134), np.uint8)

    ds2 = RasterDataset.create(
        shape=(1134, 1134),
        dtype=np.uint8,
        geoinfo=GeoInfo(
            epsg=32720,
            transform=Affine(
                10.000000005946317, 0.0, 554680.0000046354,
                0.0, -10.00000000318243, 6245339.999990689
            )
        )
    )
    ds2[:] = np.random.randint(128, 192, (1134, 1134), np.uint8)

    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [-62.403073310852044, -34.02648590051866],
            [-62.40650653839111, -34.03818674708322],
            [-62.39936113357544, -34.03943142302355],
            [-62.3962926864624, -34.02765961447532],
            [-62.403073310852044, -34.02648590051866]
        ]]
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cropped_ds, mask = ds1.crop_by_geometry(geometry, extra_ds=[ds2])
        cropped_ds.to_file(f'{tmp_dir}/cropped.png', PNG())
