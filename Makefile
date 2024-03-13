build-wheel:
	python setup.py bdist_wheel

# $1 - gdal version
# $2 - pip script version
define run_docker_test =
	echo "=> test run with GDAL=${1}"
	docker run --rm -ti -v `pwd`:/workspace -w /workspace ghcr.io/osgeo/gdal:ubuntu-small-${1} bash -c '\
		apt-get update && apt-get install -qq python3-distutils && \
		curl https://bootstrap.pypa.io/${2}get-pip.py -o /dev/stdout | python3 && \
		pip install -r requirements-dev.txt && \
		pip install -r requirements.txt && \
		pip install dataclasses future-annotations; \
		sed -i "1 s/.*/# -*- coding: future_annotations -*-/" gdal_boots/gdal.py; \
		python3 -m pytest -vv -s ./tests/; \
		sed -i "1 s/.*/from __future__ import annotations/" gdal_boots/gdal.py; \
	'
endef

test_versions = \
	docker-test/3.8.3 \
	docker-test/3.7.3 \
	docker-test/3.6.3 \
	docker-test/3.5.3 \
	docker-test/3.4.3 \
	docker-test/3.3.3 \
	docker-test/3.2.3 \
	docker-test/3.1.3 \
	docker-test/3.0.4

$(test_versions):
	$(call run_docker_test,$(shell echo '$@'|cut -d'/' -f2))

docker-test/3.0.4:
	$(call run_docker_test,$(shell echo '$@'|cut -d'/' -f2),pip/3.6/)

docker-test: $(test_versions)
	@echo "done"

upload:
	twine upload \
		--skip-existing \
		--repository-url https://pypi.onesoil.ai/ \
		-u ${ONESOIL_PYPI_USER} \
		-p ${ONESOIL_PYPI_PASSWORD} \
		dist/*


delete:
	curl -u ${ONESOIL_PYPI_USER}:${ONESOIL_PYPI_PASSWORD} \
		--form ":action=remove_pkg" \
		--form "name=gdal-boots" \
		--form "version=$(VERSION)" \
		https://pypi.onesoil.ai/
