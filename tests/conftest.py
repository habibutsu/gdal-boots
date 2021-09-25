import os

import pytest

# import warnings


# pytest.mark.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)


IMG_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture
def lena_512_png():
    return os.path.join(IMG_DIR, 'lena512color.png')


@pytest.fixture
def minsk_boundary_osm():
    return os.path.join(IMG_DIR, 'minsk-boundary.osm')


@pytest.fixture
def minsk_boundary_gpkg():
    return os.path.join(IMG_DIR, 'minsk-boundary.gpkg')


@pytest.fixture
def minsk_boundary_geojson():
    return os.path.join(IMG_DIR, 'minsk-boundary.geojson')


@pytest.fixture
def minsk_polygon():
    return {
        "type": "Polygon",
        "coordinates": [[
            [
                27.585983276367188,
                53.97284922869111
            ],
            [
                27.472000122070312,
                53.969012350740314
            ],
            [
                27.458953857421875,
                53.96517511951001
            ],
            [
                27.41809844970703,
                53.93284757750496
            ],
            [
                27.401962280273438,
                53.90211319839355
            ],
            [
                27.423934936523438,
                53.85657669031663
            ],
            [
                27.43560791015625,
                53.84746343692341
            ],
            [
                27.450714111328125,
                53.84239966092924
            ],
            [
                27.5537109375,
                53.83105458000117
            ],
            [
                27.649154663085938,
                53.83247288320114
            ],
            [
                27.66254425048828,
                53.83774044605313
            ],
            [
                27.70030975341797,
                53.87844040332883
            ],
            [
                27.698593139648438,
                53.88572576837868
            ],
            [
                27.675247192382812,
                53.94376092441113
            ],
            [
                27.670097351074215,
                53.947398072373566
            ],
            [
                27.595252990722656,
                53.97284922869111
            ],
            [
                27.585983276367188,
                53.97284922869111
            ]
        ]]
    }
