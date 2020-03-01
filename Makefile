.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

HOST:=$(shell uname -s | tr A-Z a-z)

all: build

## Conda

conda-clean:
	conda clean --all

conda-build:
	conda config --set anaconda_upload yes
	conda-build purge-all
	conda-build --user NECLA-ML recipe

## Local Development 

dev:
	git checkout dev

dev-setup: dev
	pip install -e .

uninstall-develop:
	pip uninstall $$(basename -s .git `git config --get remote.origin.url`)

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

## VCS

require-version:
ifndef version
	$(error version is undefined)
endif

clone:
	git clone --recursive $(url) $(dest)

checkout:
	git submodule update --init --recursive
	git submodule foreach -q --recursive 'git checkout $$(git config -f $$toplevel/.gitmodules submodule.$$name.branch || echo master)'

co: checkout

pull: co
	git submodule update --remote --merge --recursive
	git pull

merge:
	git checkout master
	git merge dev
	git push

tag: require-version
	git checkout master
	git tag -a v$(version) -m v$(version) 
	git push origin tags/v$(version)

release:
	git checkout $(git describe --abbrev=0 --tags)