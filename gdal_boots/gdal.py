from __future__ import annotations

import io
import logging
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from enum import Enum
from numbers import Number
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
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

from .geometry import GeometryBuilder, srs_from_epsg
from .geometry import transform as geometry_transform
from .geometry import transform_by_srs
from .options import DriverOptions, GeoJSON
from .utils import gdal_version

try:
    import orjson

    def json_dumps(data):
        return orjson.dumps(data).decode()

    json_loads = orjson.loads
except ImportError:
    import json

    json_dumps = json.dumps
    json_loads = json.loads

logger = logging.getLogger(__name__)

RawGeometry = Union[dict, ogr.Geometry]

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
if gdal_version >= (3, 7):
    DTYPE_TO_GDAL[np.int8] = gdal.GDT_Int8
GDAL_TO_DTYPE = {gdal_dtype: dtype for dtype, gdal_dtype in DTYPE_TO_GDAL.items()}

LOG_LEVELS = {
    gdal.CE_Debug: logger.debug,
    gdal.CE_None: logger.info,
    gdal.CE_Warning: logger.warning,
    gdal.CE_Failure: logger.error,
    gdal.CE_Fatal: logger.critical,
}


def error_handler(err_level, err_no, err_msg):
    LOG_LEVELS[err_level]("error_no=%s, %s", err_no, err_msg)


# gdal.PushErrorHandler('CPLQuietErrorHandler')
gdal.PushErrorHandler(error_handler)
gdal.UseExceptions()


class imdict(dict):
    """
    immutable dict
    https://www.python.org/dev/peps/pep-0351/
    """

    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


def epsg_from_srs(srs: SpatialReference) -> int:
    # value = int(srs.GetAttrValue('AUTHORITY', 1))
    value = srs.GetAuthorityCode(None)
    if not value:
        raise ValueError("Could not get epsg code")
    return int(value)


@dataclass
class GeoInfo:
    epsg: int
    transform: affine.Affine
    proj4: str = None

    @property
    def srs(self) -> SpatialReference:
        srs = SpatialReference()
        if self.epsg:
            srs.ImportFromEPSG(self.epsg)
        elif self.proj4:
            srs.ImportFromProj4(self.proj4)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return srs

    def epsg_from_srs(self, srs: SpatialReference):
        self.epsg = epsg_from_srs(srs)

    def fill_epsg_from_srs(self):
        self.epsg_from_srs(self.srs)

    def epsg_from_wkt(self, wkt):
        srs = SpatialReference()
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs.ImportFromWkt(wkt)
        self.epsg_from_srs(srs)

    @classmethod
    def from_dataset(cls, ds):
        try:
            srs = ds.GetSpatialRef()
        except AttributeError:
            # old versions
            srs = osr.SpatialReference(wkt=ds.GetProjection())

        epsg = None
        proj4 = None
        epsg_str = srs.GetAuthorityCode(None)
        if epsg_str:
            epsg = int(epsg_str)
        else:
            proj4 = srs.ExportToProj4()

        # epsg = int(srs.GetAttrValue('AUTHORITY', 1))
        return cls(epsg=epsg, transform=affine.Affine.from_gdal(*ds.GetGeoTransform()), proj4=proj4)

    def scale(self, *args):
        return type(self)(self.epsg, self.transform * affine.Affine.scale(*args))

    @property
    def projection_str(self):
        if self.epsg:
            return f"epsg:{self.epsg}"
        elif self.proj4:
            return f"proj4:{self.proj4}"
        return ""


class Resampling(Enum):
    near = "near"
    bilinear = "bilinear"
    cubic = "cubic"
    # cubic spline resampling.
    cubicspline = "cubicspline"
    # Lanczos windowed sinc resampling.
    lanczos = "lanczos"
    # average resampling, computes the weighted average of all non-NODATA contributing pixels.
    average = "average"
    # root mean square / quadratic mean of all non-NODATA contributing pixels (GDAL >= 3.3)
    rms = "rms"
    # mode resampling, selects the value which appears most often of all the sampled points.
    mode = "mode"
    # maximum resampling, selects the maximum value from all non-NODATA contributing pixels.
    max = "max"
    # minimum resampling, selects the minimum value from all non-NODATA contributing pixels.
    min = "min"
    # median resampling, selects the median value of all non-NODATA contributing pixels.
    median = "med"
    # first quartile resampling, selects the first quartile value of all non-NODATA contributing pixels.
    q1 = "q1"
    # third quartile resampling, selects the third quartile value of all non-NODATA contributing pixels.
    q3 = "q3"
    # compute the weighted sum of all non-NODATA contributing pixels (since GDAL 3.1)
    sum = "sum"


class RasterDataset:
    def __init__(self, ds: gdal.Dataset):
        self.ds = ds
        self._mem_id = None
        self.filename = None
        self.creation_options = None
        self._img = None

    @property
    def geoinfo(self) -> GeoInfo:
        return GeoInfo.from_dataset(self.ds)

    @geoinfo.setter
    def geoinfo(self, geoinfo) -> None:
        if geoinfo is not None:
            ds = self.ds
            # crs = SpatialReference()
            # crs.ImportFromEPSG(geoinfo.epsg)
            srs = geoinfo.srs
            # ds.SetSpatialRef(srs)
            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform(geoinfo.transform.to_gdal())

    @property
    def meta(self) -> dict:
        meta = self.ds.GetMetadata()
        return imdict({k: json_loads(v[5:]) if v.startswith("json:") else v for k, v in meta.items()})

    @meta.setter
    def meta(self, value: dict) -> None:
        if value:
            encoded_value = {k: f"json:{json_dumps(v)}" for k, v in value.items()}
            self.ds.SetMetadata(encoded_value)

    @property
    def shape(self) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        """
        numpy like shape
        """
        ds = self.ds
        # it's tradeoff between convenience of using and explicitness
        if ds.RasterCount == 1:
            # choose convenience
            return ds.RasterYSize, ds.RasterXSize
        return ds.RasterCount, ds.RasterYSize, ds.RasterXSize

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dtype(self):
        return GDAL_TO_DTYPE[self.ds.GetRasterBand(1).DataType]

    @property
    def resolution(self) -> Tuple[int, int]:
        return np.array([self.geoinfo.transform.a, -self.geoinfo.transform.e])

    @property
    def nodata(self):
        ds = self.ds
        return [ds.GetRasterBand(1).GetNoDataValue() for i in range(1, self.ds.RasterCount + 1)]

    @nodata.setter
    def nodata(self, value: Number):
        ds = self.ds
        if not isinstance(value, list):
            value = [value] * ds.RasterCount
        for v, i in zip(value, range(1, ds.RasterCount + 1)):
            ds.GetRasterBand(i).SetNoDataValue(v)

    def set_band_description(self, idx, description: str):
        if description:
            self.ds.GetRasterBand(idx + 1).SetDescription(description)

    def get_band_description(self, idx):
        return self.ds.GetRasterBand(idx + 1).GetDescription()

    def as_type(self, dtype) -> RasterDataset:
        ds = type(self).create(self.shape, dtype, self.geoinfo)
        ds.meta = self.meta
        ds[:] = self[:].astype(dtype)
        # copy bands descriptions
        for idx in range(self.ds.RasterCount):
            ds.set_band_description(idx, self.get_band_description(idx))
        return ds

    def __repr__(self):
        if self.ds is None:
            return f"<{type(self).__name__} {hex(id(self))} empty>"
        return (
            f"<{type(self).__name__} {hex(id(self))} {self.shape} {self.dtype.__name__} {self.geoinfo.projection_str}>"
        )

    def bounds(self, epsg=None) -> np.array:
        """
        return np.array([
            [x_min, y_min],
            [x_max, y_max]
        ])
        """
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
            geometry = GeometryBuilder().create({"type": "LineString", "coordinates": bounds})
            geometry_upd = geometry_transform(geometry, geoinfo.epsg, epsg)
            geometry.Destroy()

            bounds = geometry_upd.GetPoints(2)
            geometry_upd.Destroy()

        return np.array(bounds)

    def bounds_polygon(self, epsg=None) -> ogr.Geometry:
        [
            (min_x, min_y),
            (max_x, max_y),
        ] = self.bounds(epsg=epsg)

        polygon = GeometryBuilder().create_polygon(
            [
                [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                    (min_x, min_y),
                ]
            ]
        )
        polygon.SetCoordinateDimension(2)
        return polygon

    def set_bounds(self, coords: Iterable[Tuple[float, float]], epsg=None, resolution=None):
        """
        coords - [(xmin, ymin), (xmax, ymax)]
        epsg - projection
        resolution - x_res, y_res
        """
        x, y = np.array(coords).T
        y_size, x_size = self.shape[-2:]
        if resolution:
            res_x, res_y = resolution
        else:
            res_x = (x.max() - x.min()) / x_size
            res_y = (y.max() - y.min()) / y_size
        self.geoinfo = GeoInfo(epsg=epsg, transform=affine.Affine(res_x, 0.0, x.min(), 0.0, -res_y, y.max()))

    def __getitem__(
        self,
        slices: Union[
            int,
            slice,
            Tuple[Union[int, slice], Union[int, slice]],
            Tuple[Union[int, slice], Union[int, slice], Union[int, slice]],
        ],
    ):
        # mem_arr = self.ds.GetVirtualMemArray()
        arr = self.ds.ReadAsArray()
        return arr.__getitem__(slices)

    def __setitem__(  # noqa: C901
        self,
        selector: Union[
            int,
            slice,
            Tuple[Union[int, slice], Union[int, slice]],
            Tuple[Union[int, slice], Union[int, slice], Union[int, slice]],
        ],
        value: np.array,
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
                    raise IndexError("too many indices for array")
                bands_selector, y_selector, x_selector = selector
        else:
            bands_selector = selector

        if isinstance(bands_selector, int):
            bands_range = [bands_selector]
        elif isinstance(bands_selector, slice):
            bands_range = list(range(bands_selector.start or 0, (bands_selector.stop or ds.RasterCount)))
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
            raise NotImplementedError("not support indexing as {}".format(x_selector))

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
            raise NotImplementedError("not support indexing as {}".format(y_selector))

        ystop = ystop or ds.RasterYSize
        ysize = ystop - ystart

        if isinstance(value, Number):
            img = np.full(shape=(len(bands_range), ysize, xsize), fill_value=value)
        else:
            img = value

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

        if len(bands_range) != img.shape[0]:
            raise ValueError("could not broadcast input array")

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
    def open(cls, filename, open_flag=gdal.OF_RASTER) -> RasterDataset:
        ds = gdal.OpenEx(filename, open_flag)
        obj = cls(ds)
        obj.filename = filename
        return obj

    @classmethod
    def create(
        cls,
        shape: Union[Tuple[int, int, int], Tuple[int, int], Tuple[int, ...]],
        dtype=int,
        geoinfo: GeoInfo = None,
    ) -> RasterDataset:
        if not (2 <= len(shape) <= 3):
            raise ValueError(f"unsupported {shape=}: allowed only 2 or 3 dimensions")
        if len(shape) > 2:
            bands, height, width = shape
        else:
            bands = 1
            height, width = shape

        if isinstance(dtype, np.dtype):
            dtype = dtype.type

        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create("", width, height, bands, DTYPE_TO_GDAL[dtype])
        obj = cls(ds)
        obj.geoinfo = geoinfo
        return obj

    def to_file(self, filename: str, options: DriverOptions) -> None:
        driver = options.driver
        ds = driver.CreateCopy(filename, self.ds, strict=0, options=options.encode())
        # ds = gdal.Translate(
        #     self.filename,
        #     self._ds,
        #     format=driver.ShortName,
        #     creationOptions=self.creation_options.encode()
        # )
        ds.FlushCache()

    def is_valid(self) -> bool:
        ds = self.ds
        try:
            for i in range(1, ds.RasterCount + 1):
                ds.GetRasterBand(i).Checksum()
        except RuntimeError:
            return False
        return True

    @classmethod
    def from_stream(cls, stream: io.BytesIO, open_flag=gdal.OF_RASTER | gdal.GA_ReadOnly, ext=None) -> RasterDataset:
        mem_id = f"/vsimem/{uuid4()}"
        if ext:
            mem_id = f"{mem_id}.{ext}"

        f = gdal.VSIFOpenL(mem_id, "wb")
        for chunk in iter(lambda: stream.read(1024), b""):
            gdal.VSIFWriteL(chunk, 1, len(chunk), f)
        gdal.VSIFCloseL(f)

        ds = gdal.OpenEx(mem_id, open_flag)
        self = cls(ds)
        self._mem_id = mem_id
        return self

    def to_stream(self, stream, options):
        data = self.to_bytes(options)
        stream.write(data)

    @classmethod
    def from_bytes(cls, data: bytes, open_flag=gdal.OF_RASTER | gdal.GA_ReadOnly, ext=None) -> RasterDataset:
        mem_id = f"/vsimem/{uuid4()}"
        if ext:
            mem_id = f"{mem_id}.{ext}"
        gdal.FileFromMemBuffer(mem_id, data)
        ds = gdal.OpenEx(mem_id, open_flag)
        self = cls(ds)
        self._mem_id = mem_id
        return self

    def _to_memory(self, options: DriverOptions) -> Union[bytes, bytearray]:
        if not self.ds:
            raise ValueError("dataset was closed")

        driver = options.driver
        ext = options.driver_extensions[0]
        mem_id = f"/vsimem/{uuid4()}.{ext}"
        driver.CreateCopy(mem_id, self.ds, strict=0, options=options.encode())
        f = gdal.VSIFOpenL(mem_id, "rb")
        gdal.VSIFSeekL(f, 0, os.SEEK_END)
        size = gdal.VSIFTellL(f)
        gdal.VSIFSeekL(f, 0, 0)
        data = gdal.VSIFReadL(1, size, f)
        gdal.VSIFCloseL(f)
        # Cleanup
        gdal.Unlink(mem_id)
        return data

    def to_bytes(self, options: DriverOptions) -> bytes:
        if gdal_version >= (3, 3):
            warnings.warn("Don't use to_bytes, it's deprecated since gdal 3.3, use to_bytearray instead")
        data = self._to_memory(options)
        if isinstance(data, bytearray):
            data = bytes(data)
        return data

    def to_bytearray(self, options: DriverOptions) -> bytearray:
        data = self._to_memory(options)
        if isinstance(data, bytes):
            data = bytearray(data)
        return data

    def to_vector(self, field_id=-1, callback: Callable[[float, str, Any], None] = None) -> VectorDataset:
        """

        drv = ogr.GetDriverByName("Memory")
        feature_ds = drv.CreateDataSource("memory_name")

        https://gis.stackexchange.com/questions/328358/gdal-warp-memory-datasource-as-cutline

        """
        vds = VectorDataset.create()
        vds.add_layer("test", VectorDataset.GeometryType.Polygon, self.geoinfo.epsg)
        band = self.ds.GetRasterBand(1)
        gdal.Polygonize(band, band, vds.layers.first().ref_layer, field_id, [], callback=callback)
        vds.ds.FlushCache()
        return vds

    def _to_vector(self):
        band = self.ds.GetRasterBand(1)

        # driver = ogr.GetDriverByName('ESRI Shapefile')
        # ds = driver.CreateDataSource("test.shp")

        # driver = ogr.GetDriverByName('Memory')
        # ds = driver.CreateDataSource('memory')

        driver = ogr.GetDriverByName("GPKG")
        mem_id = f"/vsimem/{uuid4()}.gpkg"
        ds = driver.CreateDataSource(mem_id)

        layer = ds.CreateLayer("geometry", srs=self.geoinfo.srs)

        # field = ogr.FieldDefn('field', ogr.OFTInteger)
        # layer.CreateField(field)

        gdal.Polygonize(band, None, layer, -1, [], callback=None)

        ds_geom = gdal.VectorTranslate("", mem_id, format="Memory")
        # gdal.VectorTranslate('test.gpkg', ds_geom, format='GPKG')
        ds.Destroy()

        gdal.VectorTranslate("test.gpkg", ds, format="GPKG")

        return VectorDataset(ds_geom)

    def warp(  # noqa: C901
        self,
        bbox: Tuple[float, float, float, float] = None,
        bbox_epsg: int = 4326,
        bbox_srs: SpatialReference = None,
        resampling: Resampling = Resampling.near,
        extra_ds: List[RasterDataset] = None,
        resolution: Tuple[int, int] = None,
        out_epsg: int = None,
        out_proj4: str = None,
        nodata=None,
        out_nodata=None,
        width=None,
        height=None,
        cutline: VectorDataset | str = None,
    ) -> RasterDataset:
        """
        bbox: (x_min, y_min, x_max, y_max)
        """

        extra_ds = extra_ds or []
        x_res, y_res = resolution or (None, None)
        out_srs = None

        if out_proj4 and out_epsg:
            logger.warning("both parameters out_proj4 and out_epsg were specified, out_epsg will be ignored")
        if out_epsg:
            srs_obj = osr.SpatialReference()
            srs_obj.ImportFromEPSG(out_epsg)
            out_srs = srs_obj
        if out_proj4:
            out_srs = osr.SpatialReference()
            out_srs.ImportFromProj4(out_proj4)
        if not out_srs:
            out_srs = self.geoinfo.srs

        bbox_srs = bbox_srs or srs_from_epsg(bbox_epsg)

        cutlineDSName = None
        cutlineLayer = None
        tmp_file = None
        if cutline:
            if isinstance(cutline, str):
                cutlineDSName = cutline
                cutlineLayer = os.path.split(cutline)[-1].rsplit(".", maxsplit=1)[0]
            elif isinstance(cutline, VectorDataset):
                if len(cutline.layers) > 1:
                    raise ValueError("cutline should have only one layer")
                cutline_layer = cutline.layers.first()
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson")
                tmp_file_name = tmp_file.name
                tmp_file.close()
                vds = VectorDataset(cutline_layer.ref_ds)
                vds.to_file(tmp_file_name, GeoJSON())
                cutlineDSName = tmp_file_name
                cutlineLayer = os.path.split(tmp_file_name)[-1].rsplit(".", maxsplit=1)[0]
            else:
                raise ValueError("cutline should be VectorDataset or path to file")

        ds = gdal.Warp(
            "",
            [other.ds for other in extra_ds] + [self.ds],
            dstSRS=out_srs,
            xRes=x_res,
            yRes=y_res,
            outputBounds=bbox,  # (minX, minY, maxX, maxY)
            outputBoundsSRS=None if bbox is None else bbox_srs,
            resampleAlg=resampling.value,
            format="MEM",
            srcNodata=self.nodata[0] if self.nodata[0] else nodata,
            dstNodata=self.nodata[0] if self.nodata[0] else out_nodata,
            width=width,
            height=height,
            # crop to
            cutlineDSName=cutlineDSName,
            cutlineLayer=cutlineLayer,
            cropToCutline=cutlineLayer is not None,
        )
        if tmp_file:
            os.unlink(tmp_file_name)

        if ds is None:
            logger.warning("Could not warp dataset")
            return
        return type(self)(ds)

    def fast_warp_as_array(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: Tuple[int, int] = None,
    ) -> Tuple[np.array, GeoInfo]:
        """
        special case for fast sample from image

        bbox: x_min, y_min, x_max, y_max
        """
        if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError("input bbox should be in format: [x_min, y_min, x_max, y_max]")

        bounds = self.bounds()
        bbox = np.array(bbox).reshape(2, 2)
        if not (np.all(bbox[0] > bounds[0]) and np.all(bbox[1] < bounds[1])):
            raise ValueError(
                "input bbox {} should be in bounds of raster {}".format(bbox.reshape(-1), bounds.reshape(-1))
            )

        if self._img is None:
            self._img = self.ds.GetVirtualMemArray()

        img = self._img

        if resolution:
            resolution = np.array(resolution)

        ds_resolution = self.resolution

        # snap to corners
        _bbox = bbox / ds_resolution
        _bbox = np.array([np.floor(_bbox[0]), np.ceil(_bbox[1])])
        _bbox = _bbox * ds_resolution

        warp_xy = ((_bbox - bounds[0]) / ds_resolution).astype(np.uint)

        # y coordinates starts from left upper corner
        warp_xy[:, 1] = (self.shape[0] - warp_xy[:, 1])[::-1]

        epsg = self.geoinfo.epsg

        warp_img = img[slice(*warp_xy[:, 1]), slice(*warp_xy[:, 0])]

        if resolution is not None:
            raise NotImplementedError("not implemented yet")
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
                    ds_resolution[0], 0.0, _bbox[:, 0].min(), 0.0, -ds_resolution[1], _bbox[:, 1].max()
                ),
            ),
        )

    def fast_warp(
        self,
        bbox: Tuple[float, float, float, float],
        resolution: Tuple[int, int] = None,
    ) -> RasterDataset:
        warp_img, geoinfo = self.fast_warp_as_array(bbox, resolution)

        ds_warp = RasterDataset.create(shape=warp_img.shape, geoinfo=geoinfo, dtype=warp_img.dtype)
        ds_warp[:] = warp_img
        return ds_warp

    def crop_by_geometry(
        self,
        geometry: RawGeometry,
        epsg: int = 4326,
        extra_ds: List[RasterDataset] = None,
        resolution: Tuple[int, int] = None,
        out_epsg: int = None,
        out_proj4: str = None,
        resampling: Resampling = Resampling.near,
        apply_mask: bool = True,
        actual_bounds: bool = False,
    ) -> Tuple[Optional[RasterDataset], Optional[RasterDataset]]:
        """
        actual_bounds - in case when geometry much bigger actual bounds of raster
        there is no reason to make warp bigger than source file, with this
        flag you can change behaviour (default: False)
        """
        if not isinstance(geometry, ogr.Geometry):
            geometry = GeometryBuilder().create(geometry)
        extra_ds = extra_ds or []

        geom_srs = srs_from_epsg(epsg)
        ds_srs = self.geoinfo.srs
        if not geom_srs.IsSame(ds_srs):
            geometry = transform_by_srs(geometry, geom_srs, ds_srs)
            if not geometry.IsValid():
                # fix after reprojection
                geometry = geometry.MakeValid()

        if actual_bounds:
            bound_geometry = self.bounds_polygon(epsg=self.geoinfo.epsg)
            for ds in extra_ds:
                bound_geometry = bound_geometry.Union(ds.bounds_polygon(epsg=self.geoinfo.epsg))

            crop_geometry = geometry.Intersection(bound_geometry)
            json_geometry = crop_geometry.ExportToJson()
            bbox = crop_geometry.GetEnvelope()
        else:
            bbox = geometry.GetEnvelope()
            json_geometry = geometry.ExportToJson()

        vect_ds = VectorDataset.open(json_geometry, srs=ds_srs)

        # TODO: progress calback for warping
        warped_ds = self.warp(
            (bbox[0], bbox[2], bbox[1], bbox[3]),
            bbox_srs=ds_srs,
            extra_ds=extra_ds,
            resolution=resolution,
            out_epsg=out_epsg,
            out_proj4=out_proj4,
            resampling=resampling,
        )
        if warped_ds is None:
            return None, None

        mask_ds: RasterDataset = RasterDataset.create(warped_ds.shape, np.uint8, geoinfo=warped_ds.geoinfo)
        vect_ds.rasterize(mask_ds)

        if apply_mask:
            # TODO: sliding window for minizime memory usage
            # TODO: progress callback
            # TODO: use cutline
            mask_img = mask_ds[:]
            img = warped_ds[:]
            img_upd = img.copy()
            del img
            img_upd[mask_img == 0] = self.nodata[0] or 0
            warped_ds[:] = img_upd
        return warped_ds, mask_ds

    def union(self, other_ds: List[RasterDataset]) -> RasterDataset:
        geom = self.bounds_polygon()
        for ds in other_ds:
            geom = geom.Union(ds.bounds_polygon())
        x_min, x_max, y_min, y_max = geom.GetEnvelope()
        return self.warp(bbox=(x_min, y_min, x_max, y_max), bbox_epsg=self.geoinfo.epsg, extra_ds=other_ds)

    def values_by_points(self, points: List[RawGeometry]) -> list:
        if not points:
            return []

        geom_builder = GeometryBuilder()

        gt_forward = self.ds.GetGeoTransform()
        gt_reverse = gdal.InvGeoTransform(gt_forward)

        values = []
        data = self[:]
        h, w = self.shape[-2:]
        plain_raster = len(self.shape) == 2

        for point in points:
            if not isinstance(point, ogr.Geometry):
                point = geom_builder(point)
            if point.GetGeometryType() == "POINT":
                raise ValueError(f"type of geometry is not supported: {point.GetGeometryType()}")
            mx, my = point.GetX(), point.GetY()  # coord in map units

            # Convert from map to pixel coordinates
            px, py = gdal.ApplyGeoTransform(gt_reverse, mx, my)
            px = math.floor(px)  # x pixel
            py = math.floor(py)  # y pixel

            value = None
            if 0 <= px < w and 0 <= py < h:
                if plain_raster:
                    value = data[py, px]
                else:
                    value = data[:, py, px]
            values.append(value)

        return values


class Feature:
    __slots__ = ("ref_ds", "ref_feature")

    def __init__(self, ds: gdal.Dataset, feature: ogr.Feature):
        self.ref_ds = ds
        self.ref_feature = feature

    def __getitem__(self, name: str):
        return self.ref_feature.GetField(name)

    def __setitem__(self, name: str, value):
        self.ref_feature.SetField(name, value)

    def keys(self):
        return self.ref_feature.keys()

    def items(self):
        return self.ref_feature.items()

    @property
    def fid(self) -> int:
        return self.ref_feature.GetFID()

    @property
    def geometry(self) -> ogr.Geometry:
        return self.ref_feature.GetGeometryRef()

    def bounds(self):
        return self.ref_feature.GetGeometryRef().GetEnvelope()

    # TODO
    # def simplify(self, tolerance: int):
    #     self.geometry.SimplifyPreserveTopology(tolerance)


class Features:
    __slots__ = ("ref_ds", "ref_layer")

    def __init__(self, ds: gdal.Dataset, layer: ogr.Layer):
        self.ref_ds = ds
        self.ref_layer = layer

    def __len__(self) -> int:
        return self.ref_layer.GetFeatureCount()

    def __iter__(self):
        for feature in self.ref_layer:
            yield Feature(self.ref_ds, feature)

    @property
    def size(self):
        return len(self)

    def __getitem__(self, idx) -> Feature:
        feature = self.ref_layer.GetFeature(idx)
        if feature is None:
            raise IndexError(idx)
        return Feature(self.ref_ds, feature)

    def __setitem__(self, idx, feature: Feature):
        self.ref_layer.SetFeature(feature.ref_feature)


FIELD_TYPES = {
    bool: ogr.OFSTBoolean,
    int: ogr.OFTInteger,
    float: ogr.OFTReal,
    str: ogr.OFTString,
    dict: ogr.OFTString,
}

INV_FIELD_TYPES = {v: k for k, v in FIELD_TYPES.items() if k is not dict}


class Layer:
    __slots__ = ("ref_ds", "ref_layer")

    def __init__(self, ds: gdal.Dataset, layer: ogr.Layer):
        self.ref_layer = layer
        self.ref_ds = ds

    @classmethod
    def by_index(cls, ds: gdal.Dataset, idx: int):
        return cls(ds, ds.GetLayerByIndex(idx))

    @classmethod
    def by_name(cls, ds: gdal.Dataset, name: str):
        return cls(ds, ds.GetLayerByName(name))

    @property
    def name(self):
        return self.ref_layer.GetName()

    @property
    def epsg(self) -> int:
        return int(self.ref_layer.GetSpatialRef().GetAuthorityCode(None))

    def set_epsg(self, epsg: int):
        logger.warning("this is not legal way to change epsg")
        self.ref_layer.GetSpatialRef().ImportFromEPSG(epsg)

    @property
    def features(self):
        return Features(self.ref_ds, self.ref_layer)

    @property
    def field_names(self):
        layer_def: ogr.FeatureDefn = self.ref_layer.GetLayerDefn()
        fields_count = layer_def.GetFieldCount()
        result = []
        for i in range(fields_count):
            field_def: ogr.FieldDefn = layer_def.GetFieldDefn(i)
            result.append(field_def.GetName())
        return result

    @property
    def field_types(self):
        layer_def: ogr.FeatureDefn = self.ref_layer.GetLayerDefn()
        fields_count = layer_def.GetFieldCount()
        result = []
        for i in range(fields_count):
            field_def: ogr.FieldDefn = layer_def.GetFieldDefn(i)
            result.append(INV_FIELD_TYPES[field_def.GetType()])
        return result

    def add_field(self, name: str, field_type: bool, width: int = None, precision: int = None):
        field = ogr.FieldDefn(name, FIELD_TYPES[field_type])
        if field_type is dict:
            field.SetSubType(ogr.OFSTJSON)

        if width:
            field.SetWidth(width)
        if precision:
            field.SetPrecision(precision)
        self.ref_layer.CreateField(field)

    def rasterize(self, raster: RasterDataset, all_touched=True, burn_values=None):
        shape = raster.shape
        if len(shape) == 3:
            bands = list(range(1, shape[0] + 1))
            burn_values = burn_values or [1] * shape[0]
        else:
            bands = [1]
            burn_values = burn_values or [1]

        gdal.RasterizeLayer(
            raster.ds,
            bands=bands,
            layer=self.ref_layer,
            burn_values=burn_values,
            options=["ALL_TOUCHED={}".format("TRUE" if all_touched else "FALSE")],
        )

    def bounds(self, epsg=None):
        """
        return np.array([
            [x_min, y_min],
            [x_max, y_max]
        ])
        """

        x_min, y_min, x_max, y_max = self.layer.GetExtent()
        if epsg and self.epsg != epsg:
            geometry = GeometryBuilder().create(
                {
                    "type": "LineString",
                    "coordinates": [
                        # lower left
                        (x_min, y_min),
                        # upper right
                        (x_max, y_max),
                    ],
                }
            )
            geometry_upd = geometry_transform(geometry, self.epsg, epsg)
            geometry.Destroy()
            [
                # lower left
                (x_min, y_min),
                # upper right
                (x_max, y_max),
            ] = geometry_upd.GetPoints(2)
            geometry_upd.Destroy()
        return np.array([[x_min, y_min], [x_max, y_max]])

    def __repr__(self):
        return f"<{type(self).__name__} {hex(id(self))} {self.name}[{self.features.size}] epsg:{self.epsg}>"


class Layers:
    def __init__(self, ds):
        self.ds = ds

    def first(self):
        return Layer.by_index(self.ds, 0)

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            return Layer.by_name(self.ds, key)
        elif isinstance(key, int):
            return Layer.by_index(self.ds, key)
        raise ValueError("Unsuported type")

    def __iter__(self):
        def _iterator():
            for i in range(len(self)):
                yield Layer(self.ds, self.ds.GetLayerByIndex(i))

        return iter(_iterator())

    def __len__(self):
        return self.ds.GetLayerCount()

    @property
    def size(self):
        return len(self)


class VectorDataset:
    # https://livebook.manning.com/book/geoprocessing-with-python/chapter-3/1

    class GeometryType(Enum):
        Point = ogr.wkbPoint
        LineString = ogr.wkbLineString
        Polygon = ogr.wkbPolygon
        MultiPoint = ogr.wkbMultiPoint
        MultiLineString = ogr.wkbMultiLineString
        MultiPolygon = ogr.wkbMultiPolygon
        GeometryCollection = ogr.wkbGeometryCollection

    def __init__(self, ds):
        self.ds: ogr.DataSource | gdal.Dataset = ds
        # self.layers = None
        self._mem_id = None

    def __repr__(self):
        if self.ds is None:
            return f"<{type(self).__name__} {hex(id(self))} empty>"
        layers_str = ",".join([layer.name for layer in self.layers])
        return f"<{type(self).__name__} {hex(id(self))} {layers_str}>"

    # TODO
    # def fix_fid(self):
    #     gdal.VectorTranslate(
    #         layerCreationOptions=["FID=fid_fixed", "GEOMETRY_NULLABLE=NO"],
    #         SQLStatement="SELECT *, (row_number() OVER (ORDER BY 1) - 1) AS fid_fixed FROM '<layer_name>'",
    #     )

    @classmethod
    def open(cls, filename: str, open_flag=gdal.GA_ReadOnly, srs: SpatialReference = None):
        ds = gdal.OpenEx(filename, gdal.OF_VECTOR | open_flag)
        if srs:
            proj4_srs = srs.ExportToProj4()
            for idx in range(ds.GetLayerCount()):
                # this is illegal?
                ds.GetLayer(idx).GetSpatialRef().ImportFromProj4(proj4_srs)
        return cls(ds)

    @classmethod
    def create(cls):
        mem_id = f"/vsimem/{uuid4()}"
        drv: ogr.Driver = ogr.GetDriverByName("MEMORY")
        ds: ogr.DataSource = drv.CreateDataSource(mem_id)
        obj = cls(ds)
        obj._mem_id = mem_id
        # ds_ = gdal.VectorTranslate('', ds_mem, format='MEMORY', **ext_args)
        return obj

    def add_layer(self, name: str, geom_type: GeometryType, epsg: int = None) -> Layer:
        srs = None
        if epsg:
            srs = SpatialReference()
            srs.ImportFromEPSG(epsg)
        layer = self.ds.CreateLayer(name, geom_type=geom_type.value, srs=srs)
        return Layer(self.ds, layer)

    def to_file(self, filename: str, options: DriverOptions, overwrite=True) -> None:
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
        driver: ogr.Driver = ogr.GetDriverByName(options.driver_name)
        try:
            out_ds: ogr.DataSource = driver.CreateDataSource(filename)
        except RuntimeError as e:
            if overwrite:
                out_ds = None
            else:
                raise e

        if overwrite and out_ds is None:
            # if file already exists and has incorrect format
            # (for example empty) datasource will not created
            driver.DeleteDataSource(filename)
            out_ds: ogr.DataSource = driver.CreateDataSource(filename)
        else:
            raise RuntimeError(gdal.GetLastErrorMsg())

        assert out_ds is not None
        for layer in self.layers:
            out_ds.CopyLayer(layer.ref_layer, layer.name, ["OVERWRITE=YES"])

        out_ds.FlushCache()
        out_ds.Destroy()

    @classmethod
    def from_bytes(cls, data: bytes, open_flag=gdal.OF_VECTOR | gdal.GA_ReadOnly, ext=None) -> "VectorDataset":
        mem_id = f"/vsimem/{uuid4()}"
        if ext:
            mem_id = f"{mem_id}.{ext}"
        gdal.FileFromMemBuffer(mem_id, data)
        ds = gdal.OpenEx(mem_id, open_flag)
        self = cls(ds)
        self._mem_id = mem_id
        return self

    @classmethod
    def to_bytes(self, options):
        raise NotImplementedError()

    # @cached_property
    @property
    def layers(self):
        return Layers(self.ds)

    def rasterize(self, raster: RasterDataset, all_touched=True, burn_values=None):
        for layer in self.layers:
            layer.rasterize(raster, all_touched=all_touched, burn_values=burn_values)
        return raster

    def simplify(self, tolerance=5):
        for layer in self.layers:
            for feature in layer.features:
                feature.simplify(tolerance)

    def union(self, other):
        pass

    def __del__(self):
        if type(self.ds) is ogr.DataSource:
            self.ds.Destroy()
            self.ds = None
            if self._mem_id:
                # gdal.Unlink(self._mem_id)
                self._mem_id = None
        else:
            self.ds.FlushCache()
            self.ds = None
            if self._mem_id:
                gdal.Unlink(self._mem_id)
                self._mem_id = None
        self.ds = None
