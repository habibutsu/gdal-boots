import io
import json
import os
import os.path
import tempfile

import affine
import numpy as np
import pytest
import shapely.geometry
import tqdm
from osgeo import gdal
from threadpoolctl import threadpool_limits

from gdal_boots.gdal import GeoInfo, RasterDataset
from gdal_boots.geometry import GeometryBuilder
from gdal_boots.geometry import transform as geometry_transform
from gdal_boots.options import GPKG, PNG, GTiff, JP2OpenJPEG

np.random.seed(31415926)


def test_open_file(lena_512_png):
    with RasterDataset.open(lena_512_png) as ds:
        assert ds
        assert ds.shape == (3, 512, 512)
        assert ds[:, :100, :100].shape == (3, 100, 100)

        png_data = ds.to_bytes(PNG(zlevel=9))

        with tempfile.NamedTemporaryFile(suffix='.png') as fd:
            with open(fd.name, 'wb') as fd:
                fd.write(png_data)

        assert len(png_data) == 476208


def test_open_memory(lena_512_png):
    with open(lena_512_png, 'rb') as fd:
        data = fd.read()

    with RasterDataset.from_bytes(data) as ds:
        assert ds.shape == (3, 512, 512)
        assert ds[:, :100, :100].shape == (3, 100, 100)

        png_data = ds.to_bytes(PNG(zlevel=9))
        assert len(png_data) == 476208

        tiff_data = ds.to_bytes(GTiff(zlevel=9))

    with RasterDataset.from_bytes(tiff_data, open_flag=gdal.OF_RASTER | gdal.GA_Update) as ds:
        assert ds.shape

    stream = io.BytesIO(tiff_data)
    with RasterDataset.from_stream(stream, open_flag=gdal.OF_RASTER | gdal.GA_Update) as ds:
        assert ds.shape


def test_create():
    img = np.random.randint(0, 255, size=(1098, 1098), dtype=np.uint8)
    img[100:200, 100:200] = 192
    img[800:900, 800:900] = 250

    geoinfo = GeoInfo(
        epsg=32631,
        transform=affine.Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5700000.0)
    )

    with RasterDataset.create(shape=img.shape, dtype=img.dtype.type, geoinfo=geoinfo) as ds:
        ds[:, :] = img

        with tempfile.NamedTemporaryFile(suffix='.png') as fd:
            ds.to_file(fd.name, PNG())
            data = fd.read()
            assert len(data) == 1190120
            assert data[:4] == b'\x89PNG'

        with tempfile.NamedTemporaryFile(suffix='.tiff') as fd:
            ds.to_file(fd.name, GTiff())
            data = fd.read()
            assert len(data) == 1206004
            assert data[:3] == b'II*'

            assert len(ds.to_bytes(GTiff())) == len(data)

        with tempfile.NamedTemporaryFile(suffix='.jp2') as fd:
            ds.to_file(fd.name, JP2OpenJPEG())
            data = fd.read()
            assert len(data) == 303410
            assert data[:6] == b'\x00\x00\x00\x0cjP'

            assert len(ds.to_bytes(JP2OpenJPEG())) == len(data)


def test_vectorize():
    img = np.full((1098, 1098), 64, dtype=np.uint8)
    img[10:200, 10:200] = 192
    img[800:900, 800:900] = 250

    geoinfo = GeoInfo(
        epsg=32631,
        transform=affine.Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5700000.0)
    )

    with RasterDataset.create(shape=img.shape, dtype=img.dtype.type, geoinfo=geoinfo) as ds:
        ds[:, :] = img

        v_ds = ds.to_vector()
        assert v_ds
        with tempfile.NamedTemporaryFile(suffix='.gpkg') as fd:
            v_ds.to_file(fd.name, GPKG())


def test_memory():
    import json

    from osgeo import gdal, ogr
    gdal.UseExceptions()

    geojson = json.dumps({
        "type": "Point",
        "coordinates": [
            27.773437499999996,
            53.74871079689897
        ]
    })
    srcdb = gdal.OpenEx(geojson, gdal.OF_VECTOR | gdal.OF_VERBOSE_ERROR)
    # # srcdb = ogr.Open(geojson)
    # print('type', srcdb, type(srcdb))
    # gdal.VectorTranslate('test.gpkg', srcdb, format='GPKG')
    # return

    # create an output datasource in memory
    outdriver = ogr.GetDriverByName('MEMORY')
    source = outdriver.CreateDataSource('memData')

    # open the memory datasource with write access
    tmp = outdriver.Open('memData', 1)

    # copy a layer to memory
    pipes_mem = source.CopyLayer(srcdb.GetLayer(), 'pipes', ['OVERWRITE=YES'])

    # the new layer can be directly accessed via the handle pipes_mem or as source.GetLayer('pipes'):
    layer = source.GetLayer('pipes')
    for feature in layer:
        feature.SetField('SOMETHING', 1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        gdal.VectorTranslate(f'{tmp_dir}/test.gpkg', srcdb, format='GPKG')


@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/B04.tif'),
    reason='extra file "tests/fixtures/extra/B04.tif" does not exist',
)
def test_warp(minsk_polygon):
    bbox = shapely.geometry.shape(minsk_polygon).bounds

    with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
        warped_ds = ds.warp(bbox)
        assert (warped_ds.geoinfo.transform.a, -warped_ds.geoinfo.transform.e) == (10, 10)

        with tempfile.NamedTemporaryFile(suffix='.tiff') as fd:
            warped_ds.to_file(fd.name, GTiff())

        warped_ds_r100 = ds.warp(bbox, resolution=(100, 100))

        assert (warped_ds_r100.geoinfo.transform.a, -warped_ds_r100.geoinfo.transform.e) == (100, 100)
        assert all((np.array(warped_ds.shape) / 10).round() == warped_ds_r100.shape)


@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/'),
    reason='extra folder "tests/fixtures/extra/" does not exist',
)
def test_fast_warp():
    with open('tests/fixtures/35UNV_field_small.geojson') as fd:
        test_field = json.load(fd)
        geometry_4326 = GeometryBuilder.create(test_field)

    def _get_bbox(epsg):
        utm_geometry = geometry_transform(geometry_4326, 4326, epsg)
        bbox = utm_geometry.GetEnvelope()
        return np.array(bbox).reshape(2, 2).T.reshape(-1)

    with RasterDataset.open('tests/fixtures/extra/B02_10m.jp2') as ds:
        bbox = _get_bbox(ds.geoinfo.epsg)

        with tempfile.NamedTemporaryFile(prefix='10m_', suffix='.tiff') as fd:
            ds_warp = ds.fast_warp(bbox)
            ds_warp.to_file(fd.name, GTiff())

            assert ds_warp.shape == (8, 9)
            assert np.all(
                ds_warp.bounds() == np.array([[509040., 5946040.], [509130., 5946120.]])
            )
            assert ds_warp.dtype == ds.dtype

            img_warp, geoinfo = ds.fast_warp_as_array(bbox)

            assert np.all(img_warp == ds_warp[:])

    with RasterDataset.open('tests/fixtures/extra/B05_20m.jp2') as ds:
        bbox = _get_bbox(ds.geoinfo.epsg)

        with tempfile.NamedTemporaryFile(prefix='20m_', suffix='.tiff') as fd:
            ds_warp = ds.fast_warp(bbox)
            ds_warp.to_file(fd.name, GTiff())

            assert ds_warp
            assert np.all(
                ds_warp.bounds() == np.array([[509040., 5946040.], [509140., 5946120.]])
            )

    with RasterDataset.open('tests/fixtures/extra/B09_60m.jp2') as ds:
        bbox = _get_bbox(ds.geoinfo.epsg)

        with tempfile.NamedTemporaryFile(prefix='60m_', suffix='.tiff') as fd:
            ds_warp = ds.fast_warp(bbox)
            ds_warp.to_file(fd.name, GTiff())

            assert ds_warp.shape == (2, 2)
            assert np.all(
                ds_warp.bounds() == np.array([[509040., 5946000.], [509160., 5946120.]])
            )

        ds_10m = ds.warp(
            ds.bounds().reshape(-1),
            ds.geoinfo.epsg,
            resolution=(10, 10),
        )

        with tempfile.NamedTemporaryFile(prefix='60m_', suffix='.tiff') as fd:
            ds_warp = ds_10m.fast_warp(bbox)
            ds_warp.to_file(fd.name, GTiff())

            assert ds_warp.shape == (8, 9)
            assert np.all(
                ds_warp.bounds() == np.array([[509040., 5946040.], [509130., 5946120.]])
            )


@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/B04.tif'),
    reason='extra file "tests/fixtures/extra/B04.tif" does not exist',
)
def test_bounds():
    with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
        assert np.all(ds.bounds() == [
            (499980.0, 5890200.0),
            (609780.0, 6000000.0),
        ])
        assert np.all(ds.bounds(4326) == [
            (26.999700868340735, 53.16117354432605),
            (28.68033586831364, 54.136377428252246)]
                      )

    with RasterDataset.create(shape=(100, 100), dtype=np.uint8) as ds:
        ds[:] = 255
        ds[1:99, 1:99] = 0
        ds.set_bounds(
            [
                (499980.0, 5890200.0),
                (609780.0, 6000000.0),
            ],
            32635
        )
        assert np.all(ds.bounds(32635) == [
            (499980.0, 5890200.0),
            (609780.0, 6000000.0),
        ])
        ds.set_bounds(
            [
                (26.999700868340735, 53.16117354432605),
                (28.68033586831364, 54.136377428252246)
            ],
            4326
        )
        assert np.all(ds.bounds() == [
            (26.999700868340735, 53.16117354432605),
            (28.68033586831364, 54.136377428252246)
        ])
        assert np.all(ds.bounds(32635).round() == [
            [499980.0, 5890200.0],
            [609780.0, 6000000.0],
        ])


def test_crop_by_geometry():
    ds1 = RasterDataset.create(
        shape=(1134, 1134),
        dtype=np.uint8,
        geoinfo=GeoInfo(
            epsg=32720,
            transform=affine.Affine(
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
            transform=affine.Affine(
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
        cropped_ds_r100, _ = ds1.crop_by_geometry(geometry, extra_ds=[ds2], resolution=(100, 100))
        assert all((np.array(cropped_ds.shape) / 10).round() == cropped_ds_r100.shape)

    # crop by 3857
    with tempfile.TemporaryDirectory() as tmp_dir:
        geometry_4326 = GeometryBuilder.create(geometry)
        geometry_3857 = geometry_transform(geometry_4326, 4326, 3857)
        geometry_3857.FlattenTo2D()
        cropped_ds, mask = ds1.crop_by_geometry(geometry_3857, epsg=3857)
        cropped_ds.to_file(f'{tmp_dir}/cropped_by3857.tiff', GTiff())

    # crop to 3857
    with tempfile.TemporaryDirectory() as tmp_dir:
        cropped_ds_3857, mask = ds1.crop_by_geometry(geometry, out_epsg=3857)
        assert cropped_ds_3857.geoinfo.epsg == 3857
        cropped_ds_3857.to_file(f'{tmp_dir}/cropped_to3857.tiff', GTiff())


def test_write():
    img = np.ones((3, 5, 5))
    img[0] = 1
    img[1] = 2
    img[2] = 3

    ds = RasterDataset.create(shape=(3, 5, 5))
    ds[:] = 1
    ds[:] = img
    ds[0] = img[0]
    ds[:, 0] = 1
    # not supported
    # ds[:,0,:] = img[:,0]
    ds[1:3, 1:3, :] = 1
    ds[(0, 2), 2:5, 2:5] = img[(0, 2), :3, :3]

    ds = RasterDataset.create(shape=(10, 10))
    ds[2:5, 2:5] = 1


@pytest.mark.skipif(not os.getenv('TEST_COMPARE_WARP', ''), reason='skip comparison warp')
@pytest.mark.skipif(
    not os.path.exists('tests/fixtures/extra/B02_10m.jp2'),
    reason='extra file "tests/fixtures/extra/B02_10m.jp2" does not exist',
)
def test_compare_warp_fast_warp():
    np.random.randint(1622825326.8494937)

    with RasterDataset.open('tests/fixtures/extra/B02_10m.jp2') as ds:
        ds_bounds = ds.bounds()

        size = 1000
        hw_range = np.array([50, 500]) * ds.resolution

        xy = np.array([
            np.random.randint(ds_bounds[0][0], ds_bounds[1][0] - hw_range[1], size),
            np.random.randint(ds_bounds[0][1], ds_bounds[1][1] - hw_range[1], size)
        ])
        hw = np.array([
            np.random.randint(*hw_range, size),
            np.random.randint(*hw_range, size),
        ])

        bboxes = np.array([xy, xy + hw]).reshape(4, -1).T

        with threadpool_limits(limits=1, user_api='blas'):
            for bbox in tqdm.tqdm(bboxes):
                ds_warp = ds.fast_warp(bbox)

            for bbox in tqdm.tqdm(bboxes):
                img_warp, geoinfo = ds.fast_warp_as_array(bbox)

        for bbox in tqdm.tqdm(bboxes):
            ds_warp = ds.warp(bbox, bbox_epsg=ds.geoinfo.epsg)
