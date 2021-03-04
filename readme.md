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
import affine
import numpy as np
from gdal_boots.gdal import RasterDataset, GeoInfo
from gdal_boots.options import JP2OpenJPEG

img = np.random.randint(
    0, 255,
    size=(1098, 1098),
    dtype=np.uint8)

geoinfo = GeoInfo(
    epsg=32631,
    transform=affine.Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 5700000.0)
)
with RasterDataset.create(img.shape, img.dtype, geoinfo) as ds:
    ds[:,:] = img
    ds[100:500, 100:500] = 128
    ds.to_file('image.jp2', JP2OpenJPEG(quality=50))
```

usefull links:

* [GDAL/OGR Cookbook](https://pcjericks.github.io/py-gdalogr-cookbook/index.html)
* [Geoprocessing with Python / Chapter 3. Reading and writing vector data](https://livebook.manning.com/book/geoprocessing-with-python/chapter-3/157)
* [Workshop: Raster and vector processing with GDAL](http://upload.osgeo.org/gdal/workshop/foss4ge2015/workshop_gdal.pdf)
* [Advanced Python Programming for GIS / 3.9 GDAL/OGR](https://www.e-education.psu.edu/geog489/l3_p6.html)
