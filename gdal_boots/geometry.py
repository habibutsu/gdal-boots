from osgeo import ogr, osr


class GeometryBuilder:

    @classmethod
    def create(cls, geometry) -> ogr.Geometry:
        if isinstance(geometry, str):
            return ogr.CreateGeometryFromJson(geometry)

        handler = getattr(cls, 'create_{}'.format(geometry['type'].lower()))
        return handler(geometry['coordinates'])

    @classmethod
    def create_polygon(cls, coordinates) -> ogr.Geometry:
        polygon = ogr.Geometry(ogr.wkbPolygon)
        for ring_coords in coordinates:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in ring_coords:
                ring.AddPoint(*point)
            polygon.AddGeometry(ring)
        return polygon

    @classmethod
    def create_line_string(cls, coordinates) -> ogr.Geometry:
        line = ogr.Geometry(ogr.wkbLineString)
        for point in coordinates:
            line.AddPoint(*point)
        return line

    @classmethod
    def create_multipolygon(cls, coordinates) -> ogr.Geometry:
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        for polygon_coordinates in coordinates:
            polygon = cls.create_polygon(polygon_coordinates)
            multipolygon.AddGeometry(polygon)

        return multipolygon


def transform(geometry, from_epsg, to_epsg) -> ogr.Geometry:
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
