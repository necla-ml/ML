.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

HOST:=$(shell uname -s | tr A-Z a-z)

all: build

## Environment

conda-install:
	wget -O $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	sh $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh -b -p $(HOME)/miniconda3
	rm -fr $(HOME)/Downloads/Miniconda3-latest-Linux-x86_64.sh

conda-env: $(HOME)/miniconda3
	eval "`$(HOME)/miniconda3/bin/conda shell.bash hook`" && conda env create -n $(ENV) -f $(ENV).yml

conda-setup: $(HOME)/miniconda3
	echo '' >> $(HOME)/.bashrc
	echo 'eval "`$$HOME/miniconda3/bin/conda shell.bash hook`"' >> $(HOME)/.bashrc
	echo conda activate $(ENV) >> $(HOME)/.bashrc
	echo '' >> $(HOME)/.bashrc
	echo export EDITOR=vim >> $(HOME)/.bashrc
	echo export PYTHONDONTWRITEBYTECODE=1 >> $(HOME)/.bashrc

conda: conda-install conda-env conda-setup
	eval `$(HOME)/miniconda3/bin/conda shell.bash hook` && conda env list
	echo Restart your shell to create and activate conda environment "$(ENV)"

## Conda Distribution

conda-index:
	conda index /zdata/projects/shared/conda/geteigen

conda-build:
	conda config --set anaconda_upload yes
	conda-build purge-all
	conda-build --user NECLA-ML recipe

conda-clean:
	conda clean --all

## Local Development 

setup-mmdet:
	cd submodules/mmdetection; pip install -e .

dev:
	git config --global credential.helper cache --timeout=21600
	git checkout dev
	make co

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
	#git submodule foreach -q --recursive 'git checkout $$(git config -f $$toplevel/.gitmodules submodule.$$name.branch || echo master)'
	cd submodules/mmdetection; git clean -fd; git $@ -f v1.1.0
	export branch=$$(git symbolic-ref --short HEAD); git $@ $$branch

co: checkout

pull: co
	# git submodule update --remote --merge --recursive
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
