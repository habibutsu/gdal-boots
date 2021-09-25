import json

from gdal_boots.geometry import GeometryBuilder, transform


def test_create_geometry():
    geometry_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [2.295279, 48.860081],
            [2.291867, 48.857713],
            [2.293404, 48.856723],
            [2.296872, 48.859022],
            [2.295279, 48.860081]
        ]]
    }
    geometry_4326 = GeometryBuilder.create(geometry_geojson)
    geometry_4326.FlattenTo2D()

    geometry_geojson_upd = json.loads(geometry_4326.ExportToJson(options=['COORDINATE_PRECISION=6']))
    assert geometry_geojson_upd == geometry_geojson

    geometry_3857 = transform(geometry_4326, 4326, 3857)

    geometry_geojson_3857 = json.loads(geometry_3857.ExportToJson(options=['COORDINATE_PRECISION=1']))
    assert geometry_geojson_3857 == {
        'type': 'Polygon',
        'coordinates': [[
            [255509.3, 6251153.3],
            [255129.5, 6250752.7],
            [255300.6, 6250585.2],
            [255686.6, 6250974.1],
            [255509.3, 6251153.3]]
        ]
    }
