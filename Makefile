.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

HOST:=$(shell uname -s | tr A-Z a-z)
VERSION:=0.1.1

all: build

## conda

conda-clean:
	conda clean --all

build:
	# conda config --set anaconda_upload yes
	conda build .
	conda build purge

publish-osx:
	anaconda upload -u NECLA-ML --force ~/miniconda3/conda-bld/osx-64/ml-$(VERSION)-py37_0.tar.bz2

publish-linux64:
	anaconda upload -u NECLA-ML --force ~/miniconda3/conda-bld/linux-64/ml-$(VERSION)-py37_0.tar.bz2

## PIP Package Distribution

setup:
	@rm -fr dist
	python setup.py bdist_wheel

install: dist/*.whl
	pip install dist/*.whl

uninstall: dist/*.whl
	pip uninstall dist/*.whl -y

reinstall: uninstall install

clean:
	python setup.py clean --all
	@rm -fr dist

## PIP Develop

develop:
	pip install -e .

uninstall-develop:
	pip uninstall $$(basename -s .git `git config --get remote.origin.url`)

## VCS

clone:
	git clone --recursive $(url) $(dest)

checkout:
	git submodule update --init --recursive
	git submodule foreach -q --recursive 'git checkout $$(git config -f $$toplevel/.gitmodules submodule.$$name.branch || echo master)'

co: checkout

pull: co
	git submodule update --remote --merge --recursive
	git pull
