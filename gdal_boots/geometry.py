from typing import List, Sequence, Union

from osgeo import ogr, osr


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

        return handler(geometry['coordinates'])

    def create_polygon(self, coordinates: list) -> ogr.Geometry:
        polygon = ogr.Geometry(ogr.wkbPolygon)
        for ring_coords in coordinates:
            polygon.AddGeometry(self.create_linearring(ring_coords))
        return polygon

    def create_linearring(self, coordinates: List[Sequence]) -> ogr.Geometry:
        return self._add_points(ogr.Geometry(ogr.wkbLinearRing), coordinates)

    def create_linestring(self, coordinates: list) -> ogr.Geometry:
        return self._add_points(ogr.Geometry(ogr.wkbLineString), coordinates)

    def create_multipolygon(self, coordinates: list) -> ogr.Geometry:
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for polygon_coordinates in coordinates:
            multipolygon.AddGeometry(self.create_polygon(polygon_coordinates))
        return multipolygon

    def create_point(self, coordinates: list) -> ogr.Geometry:
        return self._add_point(ogr.Geometry(ogr.wkbPoint), coordinates)

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
    @classmethod
    def convert(cls, geometry: ogr.Geometry) -> dict:
        geometry_type_lower = geometry.GetGeometryName().lower()
        try:
            handler = getattr(cls, f"convert_{geometry_type_lower}")
        except AttributeError:
            raise ValueError(f"{geometry_type_lower} is not supported")
        geometry_type, coordinates = handler(geometry)
        return {
            "type": geometry_type,
            "coordinates": coordinates
        }

    @classmethod
    def convert_polygon(cls, geometry: ogr.Geometry) -> (str, dict):
        coordinates = [
            geometry.GetGeometryRef(i).GetPoints()
            for i in range(geometry.GetGeometryCount())
        ]
        return "Polygon", coordinates

    @classmethod
    def convert_multipolygon(cls, geometry: ogr.Geometry) -> (str, dict):
        coordinates = []
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            _, sub_coordinates = cls.convert_polygon(sub_geom)
            coordinates.append(sub_coordinates)
        return "MultiPolygon", coordinates

    @classmethod
    def convert_point(cls, geometry: ogr.Geometry) -> (str, dict):
        coordinates = list(geometry.GetPoints()[0])
        return "Point", coordinates


def to_geojson(geometry: ogr.Geometry, flatten: bool = True) -> dict:
    if flatten:
        geometry.FlattenTo2D()

    return GeometryGeoJson.convert(geometry)


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


def transform_geojson(geometry: dict, from_epsg: int, to_epsg: int, flatten: bool = True) -> dict:
    ogr_geometry = GeometryBuilder(flatten=flatten).create(geometry)
    new_geometry = transform(ogr_geometry, from_epsg, to_epsg)
    return to_geojson(new_geometry, flatten=False)
