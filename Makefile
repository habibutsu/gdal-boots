test_3.4:
	docker run --rm -ti -v `pwd`:/workspace -w /workspace osgeo/gdal:ubuntu-small-3.4.0 bash -c "apt-get update && \
	apt-get install -qq python3-distutils && \
	curl https://bootstrap.pypa.io/get-pip.py -o /dev/stdout | python3 && \
	pip install -r requirements-dev.txt && \
	pip install -r requirements.txt && \
	python3 -m py.test -vv -s ./tests/ \
	"

build-wheel:
	python setup.py bdist_wheel
