import json
import shapely.geometry

from gdal_boots.gdal import (
    RasterDataset,
    VectorDataset
)
from gdal_boots.options import (
    PNG,
    GTiff
)


def test_read_by_geom(minsk_polygon):

    bbox = shapely.geometry.shape(minsk_polygon).bounds

    with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
        warped_ds = ds.warp(bbox)
        vect_ds = VectorDataset.open(json.dumps(minsk_polygon))
        mask_ds = vect_ds.rasterize(warped_ds.shape, int, warped_ds.geoinfo)

        mask_img = mask_ds[:]
        img = warped_ds[:].copy()
        img[mask_img == 0] = 0
        warped_ds[:,:] = img
        warped_ds.to_file('warped_by_mask.tif', GTiff())
