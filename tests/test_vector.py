from gdal_boots.gdal import (
    VectorDataset,
    GeoInfo
)


def test_open_file(minsk_boundary_geojson):

    ds = VectorDataset.open(minsk_boundary_geojson)
    # ds.layer[0]
    # ds.layers.keys()
    # ds.layers['sdsd']
    # ds.layers.get('ddd')
    # ds.layers.first()
    # ds.layers.num(0)
    print(
        [(idx, ds.ds.GetLayerByIndex(idx).GetDescription()) for idx in range(ds.ds.GetLayerCount())]
    )

    # VectorDataset.from_string(s)