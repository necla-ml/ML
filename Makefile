.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

all: build

## PIP Package Distribution

build:
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
