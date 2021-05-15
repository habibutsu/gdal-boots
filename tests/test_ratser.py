import io
import tempfile
import affine
import shapely.geometry
import gdal
import numpy as np

from gdal_boots.gdal import (
    RasterDataset,
    GeoInfo
)

from gdal_boots.options import (
    PNG,
    GTiff,
    JP2OpenJPEG,
    GPKG
)
from gdal_boots.geometry import (
    GeometryBuilder,
    transform
)

import numpy as np

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

    with RasterDataset.from_bytes(tiff_data, open_flag=gdal.OF_RASTER|gdal.GA_Update) as ds:
        assert ds.shape

    stream = io.BytesIO(tiff_data)
    with RasterDataset.from_stream(stream, open_flag=gdal.OF_RASTER|gdal.GA_Update) as ds:
        assert ds.shape


def test_create():
    img = np.random.randint(0, 255, size=(1098, 1098), dtype=np.uint8)
    img[100:200,100:200] = 192
    img[800:900,800:900] = 250

    geoinfo = GeoInfo(
        epsg=32631,
        transform=affine.Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5700000.0)
    )

    with RasterDataset.create(shape=img.shape, dtype=img.dtype.type, geoinfo=geoinfo) as ds:
        ds[:,:] = img

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


def test_vectorize():
    img = np.full((1098, 1098), 64, dtype=np.uint8)
    img[10:200,10:200] = 192
    img[800:900,800:900] = 250

    geoinfo = GeoInfo(
        epsg=32631,
        transform=affine.Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5700000.0)
    )

    with RasterDataset.create(shape=img.shape, dtype=img.dtype.type, geoinfo=geoinfo) as ds:
        ds[:,:] = img

        v_ds = ds.to_vector()
        assert v_ds
        with tempfile.NamedTemporaryFile(suffix='.gpkg') as fd:
            v_ds.to_file(fd.name, GPKG())


def test_memory():

    from osgeo import ogr, gdal
    import json
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

    #create an output datasource in memory
    outdriver = ogr.GetDriverByName('MEMORY')
    source = outdriver.CreateDataSource('memData')

    #open the memory datasource with write access
    tmp = outdriver.Open('memData',1)

    #copy a layer to memory
    pipes_mem = source.CopyLayer(srcdb.GetLayer(), 'pipes', ['OVERWRITE=YES'])

    #the new layer can be directly accessed via the handle pipes_mem or as source.GetLayer('pipes'):
    layer = source.GetLayer('pipes')
    for feature in layer:
        feature.SetField('SOMETHING',1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        gdal.VectorTranslate(f'{tmp_dir}/test.gpkg', srcdb, format='GPKG')


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


def test_bounds():

    with RasterDataset.open('tests/fixtures/extra/B04.tif') as ds:
        assert ds.bounds() == [
            (499980.0, 5890200.0),
            (609780.0, 6000000.0),
        ]
        assert ds.bounds(4326) == [
            (26.999700868340735, 53.16117354432605),
            (28.68033586831364, 54.136377428252246)]

    with RasterDataset.create(shape=(100, 100), dtype=np.uint8) as ds:
        ds[:] = 255
        ds[1:99,1:99] = 0
        ds.set_bounds(
            [
                (499980.0, 5890200.0),
                (609780.0, 6000000.0),
            ],
            32635
        )
        assert ds.bounds(32635) == [
            (499980.0, 5890200.0),
            (609780.0, 6000000.0),
        ]
        ds.set_bounds(
            [
                (26.999700868340735, 53.16117354432605),
                (28.68033586831364, 54.136377428252246)
            ],
            4326
        )
        assert ds.bounds() == [
            (26.999700868340735, 53.16117354432605),
            (28.68033586831364, 54.136377428252246)
        ]
        assert np.array(ds.bounds(32635)).round().tolist() == [
            [499980.0, 5890200.0],
            [609780.0, 6000000.0],
        ]


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
        geometry_3857 = GeometryBuilder.create(geometry)
        transform(geometry_3857, 4326, 3857)
        # geometry_3857.FlattenTo2D()
        cropped_ds, mask = ds1.crop_by_geometry(geometry_3857, epsg=3857)
        cropped_ds.to_file(f'{tmp_dir}/cropped_by3857.tiff', GTiff())

    # crop to 3857
    with tempfile.TemporaryDirectory() as tmp_dir:
        cropped_ds_3857, mask = ds1.crop_by_geometry(geometry, out_epsg=3857)
        assert cropped_ds_3857.geoinfo.epsg == 3857
        cropped_ds.to_file(f'{tmp_dir}/cropped_to3857.tiff', GTiff())


def test_write():

    img = np.ones((3, 5, 5))
    img[0] = 1
    img[1] = 2
    img[2] = 3

    ds = RasterDataset.create(shape=(3, 5, 5))
    ds[:] = 1
    ds[:] = img
    ds[0] = img[0]
    ds[:,0] = 1
    # not supported
    # ds[:,0,:] = img[:,0]
    ds[1:3,1:3,:] = 1
    ds[(0,2), 2:5, 2:5] = img[(0, 2), :3, :3]

    ds = RasterDataset.create(shape=(10, 10))
    ds[2:5,2:5] = 1
