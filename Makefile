build-wheel:
	python setup.py bdist_wheel

ifdef GDAL_VERSION
docker-test:

	docker run --rm -ti -v `pwd`:/workspace -w /workspace osgeo/gdal:ubuntu-small-${GDAL_VERSION} bash -c '\
		apt-get update && apt-get install -qq python3-distutils && \
		curl https://bootstrap.pypa.io/${PIP_SCRIPT_VERSION}get-pip.py -o /dev/stdout | python3 && \
		pip install -r requirements-dev.txt && \
		pip install -r requirements.txt && \
		pip install dataclasses future-annotations; \
		sed -i "1 s/.*/# -*- coding: future_annotations -*-/" gdal_boots/gdal.py; \
		python3 -m pytest -vv -s ./tests/; \
		sed -i "1 s/.*/from __future__ import annotations/" gdal_boots/gdal.py; \
	'
else
docker-test:
	make docker-test GDAL_VERSION:=3.5.3
	make docker-test GDAL_VERSION:=3.4.3
	make docker-test GDAL_VERSION:=3.3.3
	make docker-test GDAL_VERSION:=3.2.3
	make docker-test GDAL_VERSION:=3.1.3
	make docker-test GDAL_VERSION:=3.0.4 PIP_SCRIPT_VERSION=pip/3.6/
endif
