import json

import pytest

from gdal_boots.geometry import GeometryBuilder, to_geojson, transform, transform_geojson


@pytest.fixture
def geometry_geojson_4326():
    return {
        "type": "Polygon",
        "coordinates": [[
            [2.295279, 48.860081],
            [2.291867, 48.857713],
            [2.293404, 48.856723],
            [2.296872, 48.859022],
            [2.295279, 48.860081]
        ]],
    }


@pytest.fixture
def geometry_4326(geometry_geojson_4326):
    geometry = GeometryBuilder.create(geometry_geojson_4326)
    geometry.FlattenTo2D()
    return geometry


def test_create_geometry(geometry_geojson_4326, geometry_4326):
    geometry_geojson_upd = json.loads(geometry_4326.ExportToJson(options=["COORDINATE_PRECISION=6"]))
    assert geometry_geojson_upd == geometry_geojson_4326


def test_transform(geometry_4326):
    geometry_3857 = transform(geometry_4326, 4326, 3857)

    geometry_geojson_3857 = json.loads(geometry_3857.ExportToJson(options=["COORDINATE_PRECISION=1"]))
    assert geometry_geojson_3857 == {
        "type": "Polygon",
        "coordinates": [[
            [255509.3, 6251153.3],
            [255129.5, 6250752.7],
            [255300.6, 6250585.2],
            [255686.6, 6250974.1],
            [255509.3, 6251153.3],
        ]],
    }


def test_transform_geojson(geometry_geojson_4326):
    geometry_geojson_3857 = transform_geojson(geometry_geojson_4326, 4326, 3857)
    assert geometry_geojson_3857 == {
        "type": "Polygon",
        "coordinates": [[
            (255509.28950849414, 6251153.3290389115),
            (255129.46740590752, 6250752.662864933),
            (255300.56546325682, 6250585.160243521),
            (255686.62145732783, 6250974.143639275),
            (255509.28950849414, 6251153.3290389115),
        ]],
    }

def test_to_geojson(geometry_geojson_4326):
    geometry = GeometryBuilder.create(geometry_geojson_4326)
    geometry_geojson = to_geojson(geometry)
    assert geometry_geojson_4326 == json.loads(json.dumps(geometry_geojson))

    geom = {
        "type": "MultiPolygon",
        "coordinates": []
    }
    assert to_geojson(GeometryBuilder.create(geom)) == geom

    geom = {
        "type": "MultiPolygon",
        "coordinates": [[]]
    }
    assert to_geojson(GeometryBuilder.create(geom)) == geom

    geom = {
        "type": "Polygon",
        "coordinates": []
    }
    assert to_geojson(GeometryBuilder.create(geom)) == geom

    geom = {
        "type": "Point",
        "coordinates": [1, 2]
    }
    assert to_geojson(GeometryBuilder.create(geom)) == geom