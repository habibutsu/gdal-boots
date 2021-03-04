# GDAL boots

Python friendly wrapper over GDAL

![boots](docs/gdalicon.png "GDAL boots")

```
"When you see someone putting on his Big Boots, you can be pretty sure that an Adventure is going to happen."

A. A. Milne
```

## WARNING

This is early implementation of library, use it for you own risk.


## Example

```python
import json
import affine
import numpy as np
from gdal_boots import RasterDataset, GeoInfo, JP2OpenJPEG, GTiff

minsk_polygon = json.load(open('./tests/fixtures/minsk-polygon.geojson'))

img = np.random.randint(
    0, 255,
    size=(1098, 1098),
    dtype=np.uint8)

geoinfo = GeoInfo(
    epsg=32635,
    transform=(
        affine.Affine(10.0, 0.0, 499980.0, 0.0, -10.0, 6000000.0) *
        affine.Affine.scale(10, 10)
    )
)
with RasterDataset.create(img.shape, img.dtype, geoinfo) as ds:
    ds[:,:] = img
    ds[100:350, 100:350] = 128
    ds.to_file('tile.jp2', JP2OpenJPEG(quality=50))

    cropped_ds = ds.crop_by_geometry(minsk_polygon)
    cropped_ds.to_file('minsk.tif', GTiff(tiled=True))
```

usefull links:

* [GDAL/OGR Cookbook](https://pcjericks.github.io/py-gdalogr-cookbook/index.html)
* [Geoprocessing with Python / Chapter 3. Reading and writing vector data](https://livebook.manning.com/book/geoprocessing-with-python/chapter-3/157)
* [Workshop: Raster and vector processing with GDAL](http://upload.osgeo.org/gdal/workshop/foss4ge2015/workshop_gdal.pdf)
* [Advanced Python Programming for GIS / 3.9 GDAL/OGR](https://www.e-education.psu.edu/geog489/l3_p6.html)
