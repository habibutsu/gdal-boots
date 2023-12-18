def get_geos_version():
    from ctypes import CDLL, c_char_p
    from ctypes.util import find_library

    _lgeos = CDLL(find_library("geos_c"))
    GEOSversion = _lgeos.GEOSversion
    GEOSversion.restype = c_char_p
    GEOSversion.argtypes = []

    return tuple(int(v) for v in GEOSversion().decode().split("-")[0].split("."))


geos_version = get_geos_version()


def get_gdal_version() -> tuple:
    try:
        from osgeo import gdal_version as osgeo_gdal_version

        return osgeo_gdal_version

    except ImportError:
        import osgeo.gdal

        raw_version = osgeo.gdal.VersionInfo()
        version_parts = []
        for idx in range(len(raw_version), 0, -2):
            version_parts.insert(0, int(raw_version[max(0, idx - 2) : idx]))
        return tuple(version_parts)


gdal_version = get_gdal_version()
