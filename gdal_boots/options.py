from dataclasses import dataclass
from enum import Enum

from osgeo import gdal

__all__ = (
    'PNG',
    'GTiff',
    'JP2OpenJPEG',
    'ECW',
    'ESRIShape',
    'GeoJSON',
    'GPKG',
)


class DriverOptions:
    def encode(self):
        return [
            ('='.join([name, str(value.value if isinstance(value, Enum) else value)])).upper()
            for name, value in vars(self).items()
            if value is not None
        ]

    @property
    def driver_name(self):
        return type(self).__name__

    @property
    def driver_extensions(self):
        driver = self.driver
        return driver.GetMetadataItem(gdal.DMD_EXTENSIONS).split(' ')

    @property
    def driver(self):
        return gdal.GetDriverByName(self.driver_name)


@dataclass
class PNG(DriverOptions):
    '''
        Portable Network Graphics
    '''

    zlevel: int = 6
    nbits: int = None


@dataclass
class GTiff(DriverOptions):
    '''
        GeoTIFF File Format
        https://gdal.org/drivers/raster/gtiff.html
    '''

    class Compress(Enum):
        lzw = 'LZW'
        jpeg = 'JPEG'
        packbits = 'PACKBITS'
        deflate = 'DEFLATE'
        zstd = 'ZSTD'
        webp = 'WEBP'
        lerc = 'LERC'
        lerc_deflate = 'LERC_DEFLATE'
        lerc_zstd = 'LERC_ZSTD'

    blockxsize: int = 256
    blockysize: int = 256
    tiled: bool = False
    compress: Compress = None
    nbits: int = None
    zlevel: int = 6


@dataclass
class JP2OpenJPEG(DriverOptions):
    '''
        JPEG2000 driver based on OpenJPEG library
        https://gdal.org/drivers/raster/jp2openjpeg.html
    '''

    quality: float = 25
    resolutions: int = None
    blockxsize: int = 1024
    blockysize: int = 1024
    nbits: int = None
    tileparts: str = 'disabled'


@dataclass
class ECW(DriverOptions):
    '''
        Enhanced Compressed Wavelets
        https://gdal.org/drivers/raster/ecw.html
    '''

    # Set the target size reduction as a percentage of the original.
    # If not provided defaults to 90% for greyscale images, and 95% for RGB images.
    # It approximates the likely ratio of input file size to output file size
    target: int = None
    ecw_format_version: int = None


@dataclass
class ESRIShape(DriverOptions):
    '''
        https://gdal.org/tutorials/vector_api_tut.html
    '''


@dataclass
class GeoJSON(DriverOptions):
    '''
        https://gdal.org/drivers/vector/geojson.html
    '''

    write_bbox = 'NO'


@dataclass
class GPKG(DriverOptions):
    '''
        GeoPackage vector
        https://gdal.org/drivers/vector/gpkg.html
    '''
