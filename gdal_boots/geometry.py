from typing import List, Sequence, Union

from osgeo import gdal, ogr, osr


class GeometryBuilder:
    def __init__(self, flatten: bool = True):
        self.flatten = flatten

    def __call__(self, geometry: Union[str, dict]) -> ogr.Geometry:
        return self.create(geometry)

    def create(self, geometry: Union[str, dict]) -> ogr.Geometry:
        if isinstance(geometry, str):
            return ogr.CreateGeometryFromJson(geometry)

        geometry_type_lower = geometry['type'].lower()
        try:
            handler = getattr(self, f"create_{geometry_type_lower}")
        except AttributeError:
            raise ValueError(f"{geometry_type_lower} is not supported")

        return handler(**geometry)

    def create_polygon(self, type:str, coordinates: list) -> ogr.Geometry:
        polygon = ogr.Geometry(ogr.wkbPolygon)
        for ring_coords in coordinates:
            polygon.AddGeometry(self.create_linearring("LinearRing", ring_coords))
        return polygon

    def create_linearring(self, type:str, coordinates: List[Sequence]) -> ogr.Geometry:
        return self._add_points(ogr.Geometry(ogr.wkbLinearRing), coordinates)

    def create_linestring(self, type:str, coordinates: list) -> ogr.Geometry:
        return self._add_points(ogr.Geometry(ogr.wkbLineString), coordinates)

    def create_multipolygon(self, type:str,  coordinates: list) -> ogr.Geometry:
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for polygon_coordinates in coordinates:
            multipolygon.AddGeometry(self.create_polygon("Polygon", polygon_coordinates))
        return multipolygon

    def create_point(self, type:str, coordinates: list) -> ogr.Geometry:
        return self._add_point(ogr.Geometry(ogr.wkbPoint), coordinates)

    def create_geometrycollection(self, type:str, geometries: List[dict]) -> ogr.Geometry:
        collection = ogr.Geometry(ogr.wkbGeometryCollection)
        for geometry in geometries:
            collection.AddGeometry(self.create(geometry))
        return collection

    def create_multilinestring(self, type:str, coordinates: List[Sequence]) -> ogr.Geometry:
        linestring = ogr.Geometry(ogr.wkbMultiLineString)
        for line_coordinates in coordinates:
            linestring.AddGeometry(self.create_linestring("LineString", line_coordinates))
        return linestring

    def _add_point(self, geometry: ogr.Geometry, point: Sequence) -> ogr.Geometry:
        if self.flatten:
            geometry.AddPoint_2D(*point)
        else:
            geometry.AddPoint(*point)
        return geometry

    def _add_points(self, geometry: ogr.Geometry, points: Sequence) -> ogr.Geometry:
        for point in points:
            self._add_point(geometry, point)
        return geometry


class GeometryGeoJson:

    def __init__(self, precision=None):
        self.precision = precision or 15

    def convert(self, geometry: ogr.Geometry) -> dict:
        geometry_type_lower = geometry.GetGeometryName().lower()
        try:
            handler = getattr(self, f"convert_{geometry_type_lower}")
        except AttributeError:
            raise ValueError(f"{geometry_type_lower} is not supported")
        return handler(geometry)


    def convert_polygon(self, geometry: ogr.Geometry) -> (str, dict):
        coordinates = [
            self._get_points(geometry.GetGeometryRef(i))
            for i in range(geometry.GetGeometryCount())
        ]
        return {
            "type": "Polygon",
            "coordinates": coordinates
        }

    def convert_multipolygon(self, geometry: ogr.Geometry) -> (str, dict):
        coordinates = []
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            sub_coordinates = self.convert_polygon(sub_geom)["coordinates"]
            coordinates.append(sub_coordinates)
        return {
            "type": "MultiPolygon",
            "coordinates": coordinates
        }

    def convert_point(self, geometry: ogr.Geometry) -> (str, dict):
        return {
            "type": "Point",
            "coordinates": self._get_points(geometry)[0]
        }

    def convert_geometrycollection(self, geometry: ogr.Geometry) -> (str, dict):
        geometries = []
        for i in range(geometry.GetGeometryCount()):
            geometries.append(self.convert(geometry.GetGeometryRef(i)))
        return {
            "type": "GeometryCollection",
            "geometries": geometries
        }

    def convert_multilinestring(self, geometry: ogr.Geometry) -> (str, dict):
        lines = []
        for i in range(geometry.GetGeometryCount()):
            lines.append(self._get_points(geometry.GetGeometryRef(i)))
        return {
            "type": "MultiLineString",
            "coordinates": lines
        }

    def _get_points(self, geometry: ogr.Geometry) -> list:
        return [
            [round(c, self.precision) for c in p]
            for p in geometry.GetPoints()
        ]


def to_geojson(geometry: ogr.Geometry, flatten: bool = True, precision=None) -> dict:
    if flatten:
        geometry.FlattenTo2D()

    return GeometryGeoJson(precision=precision).convert(geometry)


def transform(geometry: ogr.Geometry, from_epsg: int, to_epsg: int) -> ogr.Geometry:
    from_src = osr.SpatialReference()
    from_src.ImportFromEPSG(from_epsg)
    from_src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    to_crs = osr.SpatialReference()
    to_crs.ImportFromEPSG(to_epsg)
    to_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transformation = osr.CoordinateTransformation(from_src, to_crs)
    result_geometry = geometry.Clone()
    result_geometry.Transform(transformation)
    return result_geometry


def transform_geojson(geometry: dict, from_epsg: int, to_epsg: int, flatten: bool = True, precision=None) -> dict:
    ogr_geometry = GeometryBuilder(flatten=flatten).create(geometry)
    new_geometry = transform(ogr_geometry, from_epsg, to_epsg)
    return to_geojson(new_geometry, flatten=False, precision=precision)


def make_valid_geojson(geometry: dict, precision=None) -> dict:
    gdal_geometry: ogr.Geometry = GeometryBuilder().create(geometry)
    gdal_geometry = make_valid(gdal_geometry)
    return GeometryGeoJson(precision=precision).convert(gdal_geometry)


def make_valid(geometry: ogr.Geometry) -> ogr.Geometry:
    geometry.CloseRings()
    geometry = geometry.MakeValid()
    if geometry is None:
        raise RuntimeError(gdal.GetLastErrorMsg())

    if geometry.GetGeometryName() == "GEOMETRYCOLLECTION":
        union = None
        for i in range(geometry.GetGeometryCount()):
            sub_geometry = geometry.GetGeometryRef(i)

            if sub_geometry.GetGeometryName() not in ['MULTIPOLYGON', 'POLYGON']:
                continue

            if union is None:
                union = sub_geometry.Clone()
            else:
                union = union.Union(sub_geometry)
        geometry.Destroy()
        return union
    return geometry
