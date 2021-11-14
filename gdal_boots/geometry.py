from typing import Union

from osgeo import ogr, osr


class GeometryBuilder:
    @classmethod
    def create(cls, geometry: Union[str, dict]) -> ogr.Geometry:
        if isinstance(geometry, str):
            return ogr.CreateGeometryFromJson(geometry)

        geometry_type_lower = geometry['type'].lower()
        try:
            handler = getattr(cls, f"create_{geometry_type_lower}")
        except AttributeError:
            raise ValueError(f"{geometry_type_lower} is not supported")

        return handler(geometry['coordinates'])

    @classmethod
    def create_polygon(cls, coordinates: list) -> ogr.Geometry:
        polygon = ogr.Geometry(ogr.wkbPolygon)
        for ring_coords in coordinates:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in ring_coords:
                ring.AddPoint(*point)
            polygon.AddGeometry(ring)
        return polygon

    @classmethod
    def create_linearring(cls, coordinates: list) -> ogr.Geometry:
        linearring = ogr.Geometry(ogr.wkbLinearRing)
        for point in coordinates:
            linearring.AddPoint(*point)
        return linearring

    @classmethod
    def create_line_string(cls, coordinates: list) -> ogr.Geometry:
        line = ogr.Geometry(ogr.wkbLineString)
        for point in coordinates:
            line.AddPoint(*point)
        return line

    @classmethod
    def create_multipolygon(cls, coordinates: list) -> ogr.Geometry:
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        for polygon_coordinates in coordinates:
            polygon = cls.create_polygon(polygon_coordinates)
            multipolygon.AddGeometry(polygon)

        return multipolygon

    @classmethod
    def create_point(cls, coordinates: list) -> ogr.Geometry:
        point = ogr.Geometry(ogr.wkbPoint)
        if coordinates:
            point.AddPoint(*coordinates)
        return point


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
    def convert_polygon(cls, geometry: ogr.Geometry) -> dict:
        coordinates = [
            geometry.GetGeometryRef(i).GetPoints()
            for i in range(geometry.GetGeometryCount())
        ]
        return "Polygon", coordinates

    @classmethod
    def convert_multipolygon(cls, geometry: ogr.Geometry) -> dict:
        coordinates = []
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            _, sub_coordinates = cls.convert_polygon(sub_geom)
            coordinates.append(sub_coordinates)
        return "MultiPolygon", coordinates

    @classmethod
    def convert_point(cls, geometry: ogr.Geometry) -> dict:
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
    ogr_geometry = GeometryBuilder.create(geometry)
    if flatten:
        ogr_geometry.FlattenTo2D()
    new_geometry = transform(ogr_geometry, from_epsg, to_epsg)
    return to_geojson(new_geometry)


def intersection_geojson(input_geometry: dict, overlay_geometry: dict, flatten: bool = True) -> dict:
    input_geometry = GeometryBuilder.create(input_geometry)
    overlay_geometry = GeometryBuilder.create(overlay_geometry)

    if flatten:
        input_geometry.FlattenTo2D()
        overlay_geometry.FlattenTo2D()

    result = input_geometry.Intersection(overlay_geometry)

    if result.IsEmpty():
        return None

    return to_geojson(result, flatten=flatten)
