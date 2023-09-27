.PHONY: clone checkout co pull 
.PHONY: build install uninstall clean

HOST:=$(shell uname -s | tr A-Z a-z)
ARCH:=$(shell uname -m)
SHELL:=/bin/bash
CHANNEL?=NECLA-ML

all: build

## Environment


# This differs for x86 and aarch64.
ifneq ($(ARCH),aarch64)
#
# --- x86 ---
#
.PHONY: conda-install conda-env conda-setup conda
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
else
#
# --- aarch64 ---
#
# Choose a python MATCHING the Jetson pytorch build (and the OS python)
#        i.e. python 3.8 for torch-2.1 on JetPack 5.1.x
#        (We don't yet have a build38.yml file)
# a mount on snake10 with various github and jetson downloads
JETSON_DIR:=/mnt/jetson
PY_OS:=$(shell /usr/bin/python --version | cut -d' ' -f2)

#PY_VER=$(shell echo "$(PY_OS)" | awk 'BEGIN{FS="."}//{print $$1 "." $$2}')
# Actually, we have python 3.9 torch 2.0.0, which might be a better match.
# (Plan is to install the Jetson pytorch wheel and pin the version)
PY_VER:=3.9

PY_MAJ:=$(shell echo "$(PY_VER)" | awk 'BEGIN{FS="."}//{print $$1$$2}')
ENV:=build$(PY_MAJ)
CONDA:=mamba

# feedstocks Makefile is a little more generic than this one.
# Usage:     make conda; make jp51-torch
#            (or just jp51-torch, which now checks if env build$(PY_MAJ) looks OK)
# Effect:    Do whatever to install mambaforge (aarch64!) and set build env
#            Reboot (or source ~/.bashrc) to jump into conda/mamba build env
#            For Jetpack, we pin the pip-installed torch wheel in build$(PY_MAJ)
#            For quicker testing, build$(PY_MAJ) is created as a clone of base$(PY_MAJ)
# Next step: make dev-setup (for local testing)
.PHONY: conda-download0 conda-download conda-install conda-reinstall conda conda-force conda-recreate
conda-download: # for aarch64 mambabuild just did not exist.  Switch to mambaforge install.
	wget --progress=bar:force:noscroll -O $(HOME)/Downloads/Mambaforge-$$(uname)-$$(uname -m).sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$$(uname)-$$(uname -m).sh
	sh $(HOME)/Downloads/Mambaforge-$$(uname)-$$(uname -m).sh -b -p $(HOME)/miniconda3;
	rm -fr $(HOME)/Downloads/Miniconda3-latest-Linux-aarch64.sh;

conda-install: # invokes conda-download if miniconda3 is absent
	# base install is Mambaforge, so mamba command exists from outset
	if which conda; then \
		echo "Good: conda seems to be installed and active already"; \
		export | grep 'CONDA'; \
	else \
		if [ ! -d "$(HOME)/miniconda3" ]; then \
		    echo "Ohoh.  Let me install $(HOME)/miniconda3 ..."; \
		    make conda-download; \
		fi; \
		$(HOME)/miniconda3/bin/conda init bash; \
		. $(HOME)/.bashrc; \
		if which conda; then \
		    echo "Good: conda is now available"; \
		else \
		    echo "ERROR: Failed to install and activate $(HOME)/miniconda3"; \
		fi; \
		$(CONDA) upgrade -y --all; \
		$(CONDA) install -y boa -c conda-forge; \
		conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
		source $$conda_dir/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda activate base; \
		echo "make recursive: target $(MAKECMDGOALS)"; \
		$(MAKE) $(MAKECMDGOALS); \
	fi
conda-reinstall:
	rm -rf $${HOME}/miniconda3
	$(MAKE) conda-install

# base39:
#   boa : 102 pkgs, 21 MB, many things
#   anaconda-client : 29 pkgs, 52 MB
# base39 will be cloned int build39, to same time testing these scripts
conda-force: # e.g. python=3.9 --> base39, then clone into build39
	# to cut down test time, a base39 is cloned to completely reinitialize build39
	conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
	source $$conda_dir/etc/profile.d/conda.sh; \
	conda deactivate; \
	conda activate base; \
	echo "Current conda env '$$CONDA_DEFAULT_ENV'"
	if conda env list | grep "base$(PY_MAJ)"; then echo "Good: env base$(PY_MAJ) exists"; \
		else echo "Creating env base$(PY_MAJ)"; \
		conda create -n base$(PY_MAJ) -y python=$(PY_VER); fi; \
		conda list | grep boa || mamba install -y boa; \
		conda list | grep anaconda-client || mamba install -y anaconda-client; \
	if conda env list | grep "$(ENV)"; then echo "Removing old env $(ENV)"; \
		conda env remove -n "$(ENV)"; fi; \
	echo "cloning env base$(PY_MAJ) --> $(ENV)"
	conda create -n $(ENV) --clone "base$(PY_MAJ)";
	# To quickly recreate build39, conda deactivate; conda env remove -n build39; make conda

# the main conda-FOO target for aarch64...
conda:
	echo "SHELL    $(SHELL)"
	@if which conda | grep miniconda3; then \
		echo "Good. conda command is found and is miniconda3"; \
	else \
		if which conda; then \
			echo "We'll ignore a previous conda install at $${CONDA_EXE}"; \
		else \
			echo "We'll install miniconda3 now"; \
		fi; \
		$(MAKE) conda-install; \
	fi
	@if which conda; then \
		echo "Good. conda command is found."; \
	else \
		echo "Ohoh. conda command still not found."; \
		exit 1; \
	fi
	@if conda env list | grep "$(ENV)"; then \
		echo "Good: conda env $(ENV) already exists"; \
		echo "      (make conda-force to force rebuild)"; \
	else \
		echo "conda env $(ENV) not found.  Creating it..."; \
		make conda-force; \
	fi
	@# Make this the default environment...
	@if grep 'conda activate $(ENV)' $(HOME)/.bashrc; then \
	  echo 'Good. .bashrc activates $(ENV)'; \
	else \
	  echo modifying .bashrc; \
	  { \
	  echo ''; \
	  echo '# >>> feedstock $(ENV)'; \
	  echo 'conda activate $(ENV)'; \
	  echo 'export EDITOR=vim'; \
	  echo 'export PYTHONDONTWRITEBYTECODE=1'; \
	  echo '# <<< feedstock $(ENV)'; \
	  } >> $(HOME)/.bashrc; \
	fi
	echo "Current conda env '$$CONDA_DEFAULT_ENV'"; \
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: $(ENV)"; \
		else echo "Activating env $(ENV) ..."; \
		conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
		source $$conda_dir/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda activate $(ENV); \
		false; fi; \
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: $(ENV)"; \
		else echo "Trouble activating $(ENV)"; false; fi; \
	echo "Current conda env '$$CONDA_DEFAULT_ENV'";
	@echo "source ~/.bashrc yields following envs:"; \
		source ~/.bashrc && conda env list

chk00: # source .bashrc shoudl put us into $(ENV) (ex. build39)
	source ~/.bashrc && conda env list
chk-buildXX:
	@env=`source ~/.bashrc && echo $$CONDA_DEFAULT_ENV`; \
	    if [ "$$env" = "$(ENV)" ]; then : ; \
	    else echo "Oh. Let's make conda ..."; $(MAKE) conda; \
	    fi
	@# post-condition: source ~/.bashrc gets us into $(ENV)
	@env=`source ~/.bashrc && echo $$CONDA_DEFAULT_ENV`; \
	    if [ "$$env" = "$(ENV)" ]; then : ; \
	    else echo "Error ~/.bashrc does not seem to activate $(ENV)"; \
	         echo "Can you fix this manually?"; \
		 false; \
	    fi

# Usage: make jp51-torch       install jetpack-specific torch build
# TODO: wget the wheels from developer.nvidia
#       These wheels are for jetpack 5.1.x, but really should check JetPack version
#       Here we **try** to pin the pytorch version within the .bashrc environment
# This now will create 'build$(PY_MAJ)' if necessary.
.PHONY: aarch-torch-cp38 aarch-torch-cp39 aarch-torch-cp310 pin-pytorch jp51-torch
pin-pytorch:
	# tell conda to regard PyPI packages as valid substitutes for solving dependencies
	# Now can try pinning by torch=2.x.*=pypi* and it MIGHT work (does pypi have to be in the filename?)
	source ~/.bashrc; \
	conda config --env --set pip_interop_enabled True; \
	conda config --env --set channel_priority flexible; \
	pinfile="$$CONDA_PREFIX/conda-meta/pinned"; \
	if [ -f "$$pinfile" ]; then sed -i /^torch/d "$$pinfile"; fi; \
	conda list "^torch$$" | tail -n+4 | awk '{print $$1 " ==" $$2 "=pypi*"}' >> "$$pinfile"; \
	echo "pinned files in $$CONDA_PREFIX:" \
	cat "$$pinfile";

PYTORCH_DEPS:=filelock typing-extensions sympy networkx jinja2 MarkupSafe mpmath
pytorch-deps: # Prefer to have conda manage as many packages as possible...
	$(CONDA) install -y $(PYTORCH_DEPS)
aarch64-torch-cp38: chk-buildXX
	source ~/.bashrc; \
	$(MAKE) pytorch-deps; \
	pip install $(JETSON_DIR)/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl; \
	conda list torch; \
	$(MAKE) pin-pytorch
aarch64-torch-cp39: chk-buildXX pytorch-deps
	source ~/.bashrc; \
	$(MAKE) pytorch-deps; \
	pip install $(JETSON_DIR)/torch-2.0.0-1-cp39-cp39-manylinux2014_aarch64.whl; \
	conda list torch; \
	$(MAKE) pin-pytorch
aarch64-torch-cp310: chk-buildXX pytorch-deps
	source ~/.bashrc; \
	$(MAKE) pytorch-deps; \
	pip install $(JETSON_DIR)/torch-2.0.0-1-cp310-cp310-manylinux2014_aarch64.whl; \
	conda list torch; \
	$(MAKE) pin-pytorch
jp51-torch: aarch64-torch-cp$(PY_MAJ) # e.g. cp39 torch wheel
chk-torch: chk-buildXX
	@source ~/.bashrc; \
	if [ `$(CONDA) list --full-name torch | wc -l` -gt 3 ]; then \
	  echo "Good. torch is present."; \
	else \
	  echo "Oh. torch seems absent...   make jp51-torch ..."; \
	  $(MAKE) jp51-torch; \
	fi;


conda-recreate:
	rm -rf $(HOME)/miniconda3
	make conda
endif


## Conda Distribution

conda-index:
	conda index /zdata/projects/shared/conda/geteigen

conda-build:
	conda config --set anaconda_upload yes
	conda-build purge-all
	CONDA_CPUONLY_FEATURE="" \
		CONDA_CUDATOOLKIT_CONSTRAINT="    - cudatoolkit >=10.1,<10.2 # [not osx]" \
		MAGMA_PACKAGE="    - magma-cuda101 # [not osx and not win]" \
		BLD_STR_SUFFIX="" \
		conda-build --user $(CHANNEL) recipe

conda-build-cpu:
	conda config --set anaconda_upload yes
	conda-build purge-all
	CONDA_CPUONLY_FEATURE="    - cpuonly # [not osx]" \
		CONDA_CUDATOOLKIT_CONSTRAINT="    - cpuonly # [not osx]" \
		MAGMA_PACKAGE="" \
		BLD_STR_SUFFIX="_cpu" \
		conda-build --user $(CHANNEL) recipe

conda-clean:
	conda clean --all

## Local Development 

ifneq ($(ARCH),aarch64)
dev: # x86_64
	git config --global credential.helper cache --timeout=21600
	git checkout dev
	make co
else
dev: # aarch64, assume branch already created.
	git config --global credential.helper cache --timeout=21600
	git checkout orin
	make co
endif

# After a default 'make dev-setup', setup.py still has some missing things.
# Prefer to use conda versions of some requirements ...
#   NOTE:  'av' pulls in a lot, of CPU-only packages (including ffmpeg)
#   NOTE:  nvidia hw-accel ffmpeg as per nnvida jets ffmpeg via apt
#          complains about missing libnvbuf_utils.so.1.0.0, which seems
#          to want OpenCV.
#       ? sudo apt install libopencv-dev libopencv-python libopencv-samples ffmpeg ?
#   No:  this is a new issue with Jetpack 5.1.2 (could reinstall 5.1.1)
#        apparently ZED sdk from Stereolab dropped nvbuf_utils
#        Stereolabs are "working on it" (Aug 28, 2023) 
#        -- nvbuf_utils is deprecated, supposed to use NvUtils now.
ML_DEPS:=numexpr pyyaml requests requests-toolbelt av
ml-deps: # Prefer to have conda manage as many packages as possible...
	@if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: env is $(ENV)"; \
	    else echo "Activating env $(ENV) ..."; \
		conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
		source $$conda_dir/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda activate $(ENV); \
		false; fi; \
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: $(ENV)"; \
		else echo "Trouble activating $(ENV)"; false; fi; \
	$(CONDA) install -y $(ML_DEPS)

dev-setup: dev chk-torch ml-deps
	@if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: env is $(ENV)"; \
	    else echo "Activating env $(ENV) ..."; \
		conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
		source $$conda_dir/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda activate $(ENV); \
		false; fi; \
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: $(ENV)"; \
		else echo "Trouble activating $(ENV)"; false; fi; \
	echo "pip install local directory ..."; \
	pip install -vv --force-reinstall --no-deps --no-build-isolation --no-binary :all: -e .; \
	if [ -f "./test_ml_imports.py" ]; then \
		echo "testing that ml and subpackages can be imported ..."; \
		python ./test_ml_imports.py 2>&1; fi
	# import ml.av FAILS until project ml-vision is installed.

dev-clean:
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: env is $(ENV)"; \
		else echo "Activating env $(ENV) ..."; \
		conda_dir=`conda info | grep 'base environment' | awk '{print $$4}'`; \
		source $$conda_dir/etc/profile.d/conda.sh; \
		conda deactivate; \
		conda activate $(ENV); \
		false; fi; \
	if [ x"$${CONDA_DEFAULT_ENV}" = "x$(ENV)" ]; then \
		echo "Good: $(ENV)"; \
		else echo "Trouble activating $(ENV)"; false; fi; \
	pip uninstall $$(basename -s .git `git config --get remote.origin.url`); \
	python setup.py clean --all

## VCS

require-version:
ifndef version
	$(error version is undefined)
endif

clone:
	git clone --recursive $(url) $(dest)

checkout:
	git submodule update --init --recursive
	#cd submodules/mmdetection; git clean -fd; git $@ -f v2.1.0
	branch=$$(git symbolic-ref --short HEAD); \
		echo $$branch; \
		git submodule foreach -q --recursive "git checkout $$(git config -f $$toplevel/.gitmodules submodule.$$name.branch || echo $$branch)"

co: checkout

pull: co
	git submodule update --remote --merge --recursive
	git pull

merge:
	git checkout main
	git merge dev
	git push

tag: require-version
	git checkout main
	git tag -a v$(version) -m v$(version) 
	git push origin tags/v$(version)

del-tag:
	git tag -d $(tag)
	git push origin --delete tags/$(tag)

release:
	git checkout $(git describe --abbrev=0 --tags)
