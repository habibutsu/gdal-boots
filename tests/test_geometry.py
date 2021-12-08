import json

import pytest

from gdal_boots.geometry import (
    GeometryBuilder,
    to_geojson,
    transform,
    transform_geojson,
    make_valid_geojson
)


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
    geometry = GeometryBuilder().create(geometry_geojson_4326)
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
    geometry_geojson_3857 = transform_geojson(geometry_geojson_4326, 4326, 3857, precision=6)

    assert geometry_geojson_3857 == {
        "type": "Polygon",
        "coordinates": [
            [
                [255509.289508, 6251153.329039],
                [255129.467406, 6250752.662865],
                [255300.565463, 6250585.160244],
                [255686.621457, 6250974.143639],
                [255509.289508, 6251153.329039],
            ]
        ],
    }


def test_to_geojson(geometry_geojson_4326):
    geometry_builder = GeometryBuilder(flatten=True)
    geometry = geometry_builder(geometry_geojson_4326)
    geometry_geojson = to_geojson(geometry)
    assert geometry_geojson_4326 == json.loads(json.dumps(geometry_geojson))

    geom = {
        "type": "MultiPolygon",
        "coordinates": []
    }
    assert to_geojson(geometry_builder(geom)) == geom

    geom = {
        "type": "MultiPolygon",
        "coordinates": [[]]
    }
    assert to_geojson(geometry_builder(geom)) == geom

    geom = {
        "type": "Polygon",
        "coordinates": []
    }
    assert to_geojson(geometry_builder(geom)) == geom

    geom = {
        "type": "Point",
        "coordinates": [1, 2]
    }
    assert to_geojson(geometry_builder(geom)) == geom

    geom = {
        "type": "Point",
        "coordinates": [1, 2, 0]
    }
    assert to_geojson(GeometryBuilder(flatten=False).create(geom), flatten=False) == geom

    geom = {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [ 123.912278634879272, -9.949630185704605 ], [ 124.808855631265786, -9.945983073671636 ],
                        [ 124.65841639764534, -10.631108579799987 ], [ 124.419945459363774, -10.627864806741233 ],
                        [ 124.41738744984022, -10.63329355209617 ], [ 124.384577597635158, -10.780015859712114 ],
                        [ 124.378588059436339, -10.807524785302114 ], [ 124.380592550665611, -10.810794106555676 ],
                        [ 124.598579474816674, -10.901208572619954 ], [ 124.590119413296236, -10.939783777866964 ],
                        [ 123.91517943462776, -10.942551939489967 ], [ 123.912278634879272, -9.949630185704605 ]
                    ],
                    [
                        [ 124.10718673231743, -10.547923283993253 ], [ 123.983747968626105, -10.516935524633432 ],
                        [ 123.944589830279781, -10.689682964321847 ], [ 123.959685665601114, -10.694799636598699 ],
                        [ 124.149827985925143, -10.743024910106957 ], [ 124.1550455134511, -10.743005563201773 ],
                        [ 124.162079987947877, -10.716424523387417 ], [ 124.195545675010933, -10.570104692584813 ],
                        [ 124.10718673231743, -10.547923283993253 ]
                    ]
                ]
            },
            {
                "type": "MultiLineString",
                "coordinates": [
                    [
                        [ 123.91227863487927, -9.949630185704605 ], [ 124.10718673231743, -10.547923283993253 ]
                    ],
                    [
                        [ 124.10718673231743, -10.547923283993253 ], [ 124.16207998794787, -10.716424523387417 ]
                    ]
                ]
            }
        ]
    }
    geom_geojson = to_geojson(GeometryBuilder(flatten=False).create(geom), flatten=True)
    assert geom_geojson == geom


def test_make_valid():

    self_intersection = {
        "type": "Polygon",
        "coordinates": [[
            [28.377685, 53.533778],
            [28.388671, 54.278054],
            [26.768188, 53.504384],
            [26.845092, 54.226707],
            [28.377685, 53.533778]
        ]]
    }

    result = make_valid_geojson(self_intersection, precision=6)

    assert result == {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [27.582652, 53.893235],
                    [26.768188, 53.504384],
                    [26.845092, 54.226707],
                    [27.582652, 53.893235]
                ]
            ],
            [
                [
                    [27.582652, 53.893235],
                    [28.388671, 54.278054],
                    [28.377685, 53.533778],
                    [27.582652, 53.893235]
                ]
            ]
        ]
    }

    self_intersection_hole = {
        "type": "Polygon",
        "coordinates":  [[
            [26.531982, 54.204223],
            [26.740722, 53.855766],
            [28.168945, 53.448806],
            [28.372192, 54.007768],
            [26.779174, 53.402982],
            [28.883056, 53.176411],
            [28.943481, 54.188155],
            [26.531982, 54.204223]
        ]]
    }

    result = make_valid_geojson(self_intersection_hole, precision=6)

    assert result == {
        "type": "Polygon",
        "coordinates": [
            [
                [27.443987, 53.655377],
                [26.740722, 53.855766],
                [26.531982, 54.204223],
                [28.943481, 54.188155],
                [28.883056, 53.176411],
                [26.779174, 53.402982],
                [27.443987, 53.655377]
            ],
            # hole
            [
                [27.443987, 53.655377],
                [28.168945, 53.448806],
                [28.372192, 54.007768],
                [27.443987, 53.655377]
            ]
        ]
    }


    invalid = {
        "type": "MultiPolygon",
        "coordinates": [[[
            [123.912279, -9.94963],
            [124.808856, -9.945983],
            [124.658416, -10.631109],
            [124.419945, -10.627865],
            [124.417387, -10.633294],
            [124.384578, -10.780016],
            [124.378588, -10.807525],
            [124.380593, -10.810794],
            [124.598579, -10.901209],
            [124.590119, -10.939784],
            [123.915179, -10.942552],
            [123.912279, -9.94963],
            [124.16208, -10.716425],
            [124.195546, -10.570105],
            [123.983748, -10.516936],
            [123.94459, -10.689683],
            [123.959686, -10.6948],
            [124.149828, -10.743025],
            [124.155046, -10.743006],
            [124.16208, -10.716425],
            [123.912279, -9.94963]
        ]]]
    }

    result = make_valid_geojson(invalid, precision=6)

    assert result == {
        "type": "Polygon",
        "coordinates": [
            [
                [123.912279, -9.94963],
                [124.808856, -9.945983],
                [124.658416, -10.631109],
                [124.419945, -10.627865],
                [124.417387, -10.633294],
                [124.384578, -10.780016],
                [124.378588, -10.807525],
                [124.380593, -10.810794],
                [124.598579, -10.901209],
                [124.590119, -10.939784],
                [123.915179, -10.942552],
                [123.912279, -9.94963]
            ],
            [
                [124.107187, -10.547924],
                [123.983748, -10.516936],
                [123.94459, -10.689683],
                [123.959686, -10.6948],
                [124.149828, -10.743025],
                [124.155046, -10.743006],
                [124.16208, -10.716425],
                [124.195546, -10.570105],
                [124.107187, -10.547924]
            ]
        ]
    }
