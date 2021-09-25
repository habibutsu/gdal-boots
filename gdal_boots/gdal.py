import io
import os
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Tuple, Union, overload
from uuid import uuid4

import affine
import numpy as np

try:
    from functools import cached_property
except ImportError:
    from functools import lru_cache


    def cached_property(fn):

        @property
        @lru_cache(maxsize=None)
        def wrapper(self):
            return fn(self)

        return wrapper

from osgeo import gdal, ogr, osr
from osgeo.osr import SpatialReference

from .geometry import GeometryBuilder
from .geometry import transform as geometry_transform

try:
    import orjson as json
except ImportError:
    import json

DTYPE_TO_GDAL = {
    np.uint8: gdal.GDT_Byte,
    np.uint16: gdal.GDT_UInt16,
    np.uint32: gdal.GDT_UInt32,
    np.float32: gdal.GDT_Float32,
    np.int16: gdal.GDT_Int16,
    np.int32: gdal.GDT_Int32,
    np.float64: gdal.GDT_Float64,
    int: gdal.GDT_Int32,
    float: gdal.GDT_Float64,
}
GDAL_TO_DTYPE = {
    gdal_dtype: dtype
    for dtype, gdal_dtype in DTYPE_TO_GDAL.items()
}
gdal.UseExceptions()


@dataclass
class GeoInfo:
    epsg: int
    transform: affine.Affine

    @property
    def srs(self):
        srs = SpatialReference()
        srs.ImportFromEPSG(self.epsg)
        return srs

    def epsg_from_wkt(self, wkt):
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        # int(srs.GetAttrValue('AUTHORITY',1))

        self.epsg = int(srs.GetAuthorityCode(None))

    @classmethod
    def from_dataset(cls, ds):
        try:
            srs = ds.GetSpatialRef()
        except AttributeError:
            # old versions
            srs = osr.SpatialReference(wkt=ds.GetProjection())

        epsg = int(srs.GetAuthorityCode(None))
        # epsg = int(srs.GetAttrValue('AUTHORITY', 1))
        return cls(
            epsg=epsg,
            transform=affine.Affine.from_gdal(*ds.GetGeoTransform())
        )


class Resampling(Enum):
    near = 'near'
    bilinear = 'bilinear'
    cubic = 'cubic'
    # cubic spline resampling.
    cubicspline = 'cubicspline'
    # Lanczos windowed sinc resampling.
    lanczos = 'lanczos'
    # average resampling, computes the weighted average of all non-NODATA contributing pixels.
    average = 'average'
    # root mean square / quadratic mean of all non-NODATA contributing pixels (GDAL >= 3.3)
    rms = 'rms'
    # mode resampling, selects the value which appears most often of all the sampled points.
    mode = 'mode'
    # maximum resampling, selects the maximum value from all non-NODATA contributing pixels.
    max = 'max'
    # minimum resampling, selects the minimum value from all non-NODATA contributing pixels.
    min = 'min'
    # median resampling, selects the median value of all non-NODATA contributing pixels.
    median = 'med'
    # first quartile resampling, selects the first quartile value of all non-NODATA contributing pixels.
    q1 = 'q1'
    # third quartile resampling, selects the third quartile value of all non-NODATA contributing pixels.
    q3 = 'q3'
    # compute the weighted sum of all non-NODATA contributing pixels (since GDAL 3.1)
    sum = 'sum'


class RasterDataset:

    def __init__(self, ds: gdal.Dataset):
        self.ds = ds
        self._mem_id = None
        self.filename = None
        self.creation_options = None
        self._img = None

    @property
    def geoinfo(self):
        return GeoInfo.from_dataset(self.ds)

    @geoinfo.setter
    def geoinfo(self, geoinfo):
        if geoinfo is not None:
            ds = self.ds
            # crs = SpatialReference()
            # crs.ImportFromEPSG(geoinfo.epsg)
            ds.SetSpatialRef(geoinfo.srs)
            ds.SetGeoTransform(geoinfo.transform.to_gdal())

    @property
    def meta(self):
        return self.ds.GetMetadata()

    @meta.setter
    def meta(self, value: dict):
        if value:
            self.ds.SetMetadata(*['{}={}'.format(k, v) for k, v in value.items()])

    @property
    def shape(self):
        ds = self.ds
        # it's tradeoff between convenience of using and explicitness
        if ds.RasterCount == 1:
            # choose convenience
            return (ds.RasterYSize, ds.RasterXSize)
        return (ds.RasterCount, ds.RasterYSize, ds.RasterXSize)

    @property
    def dtype(self):
        return GDAL_TO_DTYPE[self.ds.GetRasterBand(1).DataType]

    @property
    def resolution(self):
        return np.array([self.geoinfo.transform.a, -self.geoinfo.transform.e])

    def as_type(self, dtype):
        ds = type(self).create(self.shape, dtype, self.geoinfo)
        ds.meta = self.meta
        ds[:] = self[:].astype(dtype)
        return ds

    # def __repr__(self):
    #     return f'<{type(self).__name__} {hex(id(self))} {self.shape} {self.dtype.__name__} epsg:{self.geoinfo.epsg}>'

    def bounds(self, epsg=None) -> np.array:
        '''
            return np.array([
                [x_min, y_min],
                [x_max, y_max]
            ])
        '''
        # gcps = self.ds.GetGCPs()
        # [[gcp.GCPX, gcp.GCPY] for gcp in gcps]

        shape = self.shape
        if len(shape) == 2:
            y_size, x_size = shape
        else:
            _, y_size, x_size = shape
        geoinfo = self.geoinfo
        transform = geoinfo.transform

        xb1 = transform.c
        yb1 = transform.f
        xb2 = xb1 + transform.a * x_size
        yb2 = yb1 + transform.e * y_size

        min_x = min(xb1, xb2)
        min_y = min(yb1, yb2)
        max_x = max(xb1, xb2)
        max_y = max(yb1, yb2)
        bounds = [
            # lower left
            (min_x, min_y),
            # upper right
            (max_x, max_y),
        ]
        if epsg and epsg != geoinfo.epsg:
            geometry = GeometryBuilder.create_line_string(bounds)
            geometry_upd = geometry_transform(geometry, geoinfo.epsg, epsg)
            geometry.Destroy()

            bounds = geometry_upd.GetPoints(2)
            geometry_upd.Destroy()

        return np.array(bounds)

    def bounds_polygon(self, epsg=None):
        [
            (min_x, min_y),
            (max_x, max_y),
        ] = self.bounds(epsg=epsg)

        polygon = GeometryBuilder.create_polygon([[
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]])
        polygon.SetCoordinateDimension(2)
        return polygon

    def set_bounds(self, coords, epsg=None, resolution=None):
        x, y = np.array(coords).T
        x_size, y_size = self.shape
        res_x = (x.max() - x.min()) / x_size
        rex_y = (y.max() - y.min()) / y_size
        self.geoinfo = GeoInfo(
            epsg=epsg,
            transform=affine.Affine(
                res_x, 0.0, x.min(),
                0.0, -rex_y, y.max()
            )
        )

    def __getitem__(self, slices: Tuple[slice, slice, slice]):
        # mem_arr = self.ds.GetVirtualMemArray()
        arr = self.ds.ReadAsArray()
        return arr.__getitem__(slices)

    def __setitem__(
        self,
        selector: Union[
            int,
            slice,
            Tuple[Union[int, slice], Union[int, slice]],
            Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]
        ],
        value: np.array
    ):
        ds = self.ds
        x_selector = None
        y_selector = None
        shape = self.shape
        # match
        if isinstance(selector, tuple):
            if len(selector) == 2:
                # implicitly selection
                if len(shape) == 3:
                    bands_selector, y_selector = selector
                else:
                    bands_selector = 0
                    y_selector, x_selector = selector
            elif len(selector) == 3:
                if len(shape) == 2:
                    raise IndexError('too many indices for array')
                bands_selector, y_selector, x_selector = selector
        else:
            bands_selector = selector

        if isinstance(bands_selector, int):
            bands_range = [bands_selector]
        elif isinstance(bands_selector, slice):
            bands_range = list(
                range(
                    bands_selector.start or 0,
                    (bands_selector.stop or ds.RasterCount)
                )
            )
        elif isinstance(bands_selector, (tuple, list)):
            bands_range = bands_selector

        xstop = None
        if x_selector is None:
            xstart = 0
        elif isinstance(x_selector, int):
            xstart = x_selector
            xstop = xstart + 1
        elif isinstance(x_selector, slice):
            xstart = x_selector.start or 0
            xstop = x_selector.stop
        else:
            raise NotImplementedError('not support indexing as {}'.format(x_selector))

        xstop = xstop or ds.RasterXSize
        xsize = xstop - xstart

        ystop = None
        if y_selector is None:
            ystart = 0
        elif isinstance(y_selector, int):
            ystart = y_selector
            ystop = ystart + 1
        elif isinstance(y_selector, slice):
            ystart = y_selector.start or 0
            ystop = y_selector.stop
        else:
            raise NotImplementedError('not support indexing as {}'.format(y_selector))

        ystop = ystop or ds.RasterYSize
        ysize = ystop - ystart

        if isinstance(value, Number):
            img = np.full(
                shape=(len(bands_range), ysize, xsize),
                fill_value=value
            )
        else:
            img = value

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        if len(bands_range) != img.shape[0]:
            raise ValueError('could not broadcast input array')

        for in_band_num, band_num in enumerate(bands_range):
            band = ds.GetRasterBand(band_num + 1)
            band.WriteArray(img[in_band_num], xstart, ystart)

    def add_band(self, value: np.array = None):
        ds = self.ds
        ds.AddBand()
        if value:
            band = ds.GetRasterBand(ds.RasterCount)
            band.WriteArray(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.__del__()

    def __del__(self):
        if self.ds:
            self.ds.FlushCache()
        if self._mem_id:
            gdal.Unlink(self._mem_id)
            self._mem_id = None
        self.ds = None

    @classmethod
    def open(cls, filename, open_flag=gdal.OF_RASTER):

        ds = gdal.OpenEx(filename, open_flag)
        obj = cls(ds)
        obj.filename = filename
        return obj

    @classmethod
    def create(cls, shape: Union[Tuple[int, int, int], Tuple[int, int]], dtype=int, geoinfo: GeoInfo = None):
        if len(shape) > 2:
            bands, height, width = shape
        else:
            bands = 1
            height, width = shape

        if isinstance(dtype, np.dtype):
            dtype = dtype.type

        driver = gdal.GetDriverByName('MEM')
        ds = driver.Create('', width, height, bands, DTYPE_TO_GDAL[dtype])
        self = cls(ds)
        self.geoinfo = geoinfo
        return self

    def to_file(self, filename, options=None):
        driver = options.driver
        ds = driver.CreateCopy(
            filename,
            self.ds,
            strict=0, options=options.encode()
        )
        # ds = gdal.Translate(
        #     self.filename,
        #     self._ds,
        #     format=driver.ShortName,
        #     creationOptions=self.creation_options.encode()
        # )
        ds.FlushCache()

    @classmethod
    def from_stream(cls, stream: io.BytesIO, open_flag=gdal.OF_RASTER | gdal.GA_ReadOnly, ext=None):
        mem_id = f'/vsimem/{uuid4()}'
        if ext:
            mem_id = f'{mem_id}.{ext}'

        f = gdal.VSIFOpenL(mem_id, 'wb')
        for chunk in iter(lambda: stream.read(1024), b''):
            gdal.VSIFWriteL(chunk, 1, len(chunk), f)
        gdal.VSIFCloseL(f)

        ds = gdal.OpenEx(mem_id, open_flag)
        self = cls(ds)
        self._mem_id = mem_id
        return self

    @classmethod
    def from_bytes(cls, data, open_flag=gdal.OF_RASTER | gdal.GA_ReadOnly, ext=None):
        mem_id = f'/vsimem/{uuid4()}'
        if ext:
            mem_id = f'{mem_id}.{ext}'
        gdal.FileFromMemBuffer(mem_id, data)
        ds = gdal.OpenEx(mem_id, open_flag)
        self = cls(ds)
        self._mem_id = mem_id
        return self

    def to_bytes(self, options):
        if not self.ds:
            raise ValueError('dataset was closed')

        driver = options.driver
        ext = options.driver_extensions[0]
        mem_id = f'/vsimem/{uuid4()}.{ext}'
        driver.CreateCopy(
            mem_id,
            self.ds,
            strict=0,
            options=options.encode()
        )
        f = gdal.VSIFOpenL(mem_id, 'rb')
        gdal.VSIFSeekL(f, 0, os.SEEK_END)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        data = gdal.VSIFReadL(1, size, f)
        gdal.VSIFCloseL(f)
        # Cleanup
        gdal.Unlink(mem_id)
        return data

    def to_vector(self):
        '''
        drv = ogr.GetDriverByName("Memory")
        feature_ds = drv.CreateDataSource("memory_name")

        https://gis.stackexchange.com/questions/328358/gdal-warp-memory-datasource-as-cutline

        '''
        band = self.ds.GetRasterBand(1)
        mem_id = f'/vsimem/{uuid4()}'
        ds = ogr.GetDriverByName('MEMORY').CreateDataSource(mem_id)
        layer = ds.CreateLayer('geometry', srs=self.geoinfo.srs)
        gdal.Polygonize(band, None, layer, -1, [], callback=None)
        ds.FlushCache()

        return VectorDataset(ds)

    def _to_vector(self):
        band = self.ds.GetRasterBand(1)

        # driver = ogr.GetDriverByName('ESRI Shapefile')
        # ds = driver.CreateDataSource("test.shp")

        # driver = ogr.GetDriverByName('Memory')
        # ds = driver.CreateDataSource('memory')

        driver = ogr.GetDriverByName('GPKG')
        mem_id = f'/vsimem/{uuid4()}.gpkg'
        ds = driver.CreateDataSource(mem_id)

        layer = ds.CreateLayer('geometry', srs=self.geoinfo.srs)

        # field = ogr.FieldDefn('field', ogr.OFTInteger)
        # layer.CreateField(field)

        gdal.Polygonize(band, None, layer, -1, [], callback=None)

        ds_geom = gdal.VectorTranslate('', mem_id, format='Memory')
        # gdal.VectorTranslate('test.gpkg', ds_geom, format='GPKG')
        ds.Destroy()

        gdal.VectorTranslate('test.gpkg', ds, format='GPKG')

        return VectorDataset(ds_geom)

    def warp(
        self,
        bbox,
        bbox_epsg=4326,
        resampling=Resampling.near, extra_ds=[],
        resolution=None,
        out_epsg=None
    ):
        '''
            bbox - x_min, y_min, x_max, y_max
        '''
        x_res, y_res = resolution if resolution else (None, None)
        ds = gdal.Warp('',
                       [other.ds for other in extra_ds] + [self.ds],
                       dstSRS=f'epsg:{out_epsg}' if out_epsg else self.geoinfo.srs,
                       xRes=x_res or self.geoinfo.transform.a,
                       yRes=y_res or -self.geoinfo.transform.e,
                       outputBounds=bbox,  # (minX, minY, maxX, maxY)
                       outputBoundsSRS='epsg:{}'.format(bbox_epsg),
                       resampleAlg=resampling.value,
                       format="MEM"
                       )
        return type(self)(ds)

    def fast_warp_as_array(self, bbox, resolution=None) -> Tuple[np.array, GeoInfo]:
        '''
            special case for fast sample from image

            bbox: x_min, y_min, x_max, y_max
        '''
        if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError('input bbox should be in format: [x_min, y_min, x_max, y_max]')

        bounds = self.bounds()
        bbox = np.array(bbox).reshape(2, 2)
        if not (np.all(bbox[0] > bounds[0]) and np.all(bbox[1] < bounds[1])):
            raise ValueError('input bbox {} should be in bounds of raster {}'.format(
                bbox.reshape(-1), bounds.reshape(-1)))

        if self._img is None:
            self._img = self.ds.GetVirtualMemArray()

        img = self._img

        if resolution:
            resolution = np.array(resolution)

        ds_resolution = self.resolution

        # snap to corners
        _bbox = (bbox / ds_resolution)
        _bbox = np.array([np.floor(_bbox[0]), np.ceil(_bbox[1])])
        _bbox = (_bbox * ds_resolution)

        warp_xy = ((_bbox - bounds[0]) / ds_resolution).astype(np.uint)

        # y coordinates starts from left upper corner
        warp_xy[:, 1] = (self.shape[0] - warp_xy[:, 1])[::-1]

        epsg = self.geoinfo.epsg

        warp_img = img[slice(*warp_xy[:, 1]), slice(*warp_xy[:, 0])]

        if resolution is not None:
            raise NotImplementedError('not implemented yet')
            # k = ds_resolution / resolution
            # warp_img = np.repeat(np.repeat(warp_img, k[0], axis=0), k[1], axis=1)
            # # so recalulate bbox and crop img to it

            # _bbox_upd = (bbox / resolution)
            # _bbox_upd = np.array([np.floor(_bbox_upd[0]), np.ceil(_bbox_upd[1])])
            # _bbox_upd = (_bbox_upd * resolution).astype(np.uint)

            # crop_xy = (np.array(warp_img.shape) - (((_bbox - _bbox_upd) / resolution)[1][::-1])).astype(np.uint)
            # warp_img = warp_img[:crop_xy[0], :crop_xy[1]]
            # ds_resolution = resolution
            # _bbox = _bbox_upd

        return (
            warp_img,
            GeoInfo(
                epsg=epsg,
                transform=affine.Affine(
                    ds_resolution[0], 0.0, _bbox[:, 0].min(),
                    0.0, -ds_resolution[1], _bbox[:, 1].max()
                )
            )
        )

    def fast_warp(self, bbox, resolution=None) -> 'RasterDataset':
        warp_img, geoinfo = self.fast_warp_as_array(bbox, resolution)

        ds_warp = RasterDataset.create(
            shape=warp_img.shape,
            geoinfo=geoinfo,
            dtype=warp_img.dtype
        )
        ds_warp[:] = warp_img
        return ds_warp

    def crop_by_geometry(
        self,
        geometry: Union[dict, ogr.Geometry],
        epsg=4326,
        extra_ds=[],
        resolution=None,
        out_epsg=None,
        resampling=Resampling.near,
        apply_mask=True,
    ):
        if not isinstance(geometry, ogr.Geometry):
            geometry = GeometryBuilder.create(geometry)

        bbox = geometry.GetEnvelope()
        warped_ds = self.warp(
            (bbox[0], bbox[2], bbox[1], bbox[3]),
            bbox_epsg=epsg,
            extra_ds=extra_ds,
            resolution=resolution,
            out_epsg=out_epsg,
            resampling=resampling
        )
        vect_ds = VectorDataset.open(geometry.ExportToJson())
        if epsg != 4326:
            vect_ds.ds.GetLayer(0).GetSpatialRef().ImportFromEPSG(epsg)

        mask_ds = RasterDataset.create(warped_ds.shape, warped_ds.dtype, geoinfo=warped_ds.geoinfo)
        vect_ds.rasterize(mask_ds)
        mask_img = mask_ds[:]
        if apply_mask:
            img = warped_ds[:].copy()
            img[mask_img == 0] = 0
            warped_ds[:] = img
        return warped_ds, mask_ds


class Feature:

    def __init__(self, feature):
        self.feature = feature

    def __getitem__(self, name: str):
        return self.feature.items()[name]

    def keys(self):
        return self.feature.keys()

    @property
    def geometry(self):
        return self.feature.geometry()


class Features:

    def __init__(self, layer):
        self.layer = layer

    def __len__(self):
        return self.layer.GetFeatureCount()

    def __getitem__(self, idx):
        return Feature(self.layer.GetFeature(idx))


class Layer:

    def __init__(self, layer):
        self.layer = layer

    @property
    def name(self):
        return self.layer.GetName()

    @property
    def features(self):
        return Features(self.layer)


class Layers:

    def __init__(self, ds):
        self.ds = ds

    def first(self):
        return Layer(self.ds.GetLayerByIndex(0))

    def __getitem__(self, idx):
        return Layer(self.ds.GetLayerByIndex(idx))

    def __iter__(self):
        def _iterator():
            for i in range(len(self)):
                yield Layer(self.ds.GetLayerByIndex(i))

        return iter(_iterator())

    def __len__(self):
        return self.ds.GetLayerCount()


class VectorDataset:

    # https://livebook.manning.com/book/geoprocessing-with-python/chapter-3/1

    def __init__(self, ds):
        self.ds = ds
        # self.layers = None
        self._mem_id = None

    @classmethod
    def open(cls, filename, open_flag=gdal.GA_ReadOnly):
        ds = gdal.OpenEx(filename, gdal.OF_VECTOR | open_flag)
        return cls(ds)

    @classmethod
    def create(cls):
        mem_id = f'/vsimem/{uuid4()}'
        drv = ogr.GetDriverByName('MEMORY')
        ds = drv.CreateDataSource(mem_id)

        ds.CreateLayer()

        # gdal.FileFromMemBuffer(mem_id, data)
        # ds = gdal.OpenEx(mem_id, gdal.GA_ReadOnly)
        self = cls(ds)
        self._mem_id = mem_id
        # ds_ = gdal.VectorTranslate('', ds_mem, format='MEMORY', **ext_args)
        return self

    def to_file(self, filename, options):
        # # No such file or directory
        # gdal.VectorTranslate('', mem_id, format='MEMORY')
        # gdal.VectorTranslate(filename, self.ds, format='GPKG')
        # gdal.VectorTranslate('', df_geom_str, format='Memory', srcSRS=to_crs, dstSRS=to_crs)

        # out_ds = ogr.GetDriverByName('GPKG').CreateDataSource('copied_ds.gpkg')
        # out_ds.CopyLayer(ds.GetLayer(), 'geometry', ['OVERWRITE=YES'])
        # out_ds.FlushCache()

        # srcdb = gdal.OpenEx(mem_id, gdal.OF_VECTOR | gdal.OF_VERBOSE_ERROR)
        # gdal.VectorTranslate('ds2.gpkg', srcdb, format='GPKG')

        # # field = ogr.FieldDefn('field', ogr.OFTInteger)
        # # layer.CreateField(field)

        driver = ogr.GetDriverByName(options.driver_name)
        out_ds = driver.CreateDataSource(filename)
        if out_ds is None:
            # if file already exists and has incorrect format
            # (for example empty) datasource will not created
            driver.DeleteDataSource(filename)
        out_ds = driver.CreateDataSource(filename)

        out_ds.CopyLayer(self.ds.GetLayer(), 'geometry', ['OVERWRITE=YES'])
        out_ds.FlushCache()
        out_ds.Destroy()

    @classmethod
    def from_string(cls, value):
        # driver = ogr.GetDriverByName('Memory')
        # self.ds = driver.CreateDataSource('memory')
        # gdal.VectorTranslate('', df_geom_str, format='MEMORY', srcSRS=to_crs, dstSRS=to_crs)
        pass

    @classmethod
    def to_string(self):
        pass

    # @cached_property
    @property
    def layers(self):
        return Layers(self.ds)

    def rasterize(self, raster, all_touched=True):
        shape = raster.shape
        if len(shape) == 3:
            bands = list(range(1, shape[0] + 1))
            burn_values = [1] * shape[0]
        else:
            bands = [1]
            burn_values = [1]

        for layer in self.layers:
            gdal.RasterizeLayer(
                raster.ds,
                bands=bands,
                layer=layer.layer,
                burn_values=burn_values,
                options=['ALL_TOUCHED={}'.format('TRUE' if all_touched else 'FALSE')]
            )
        return raster

    def simplify(self):
        pass
