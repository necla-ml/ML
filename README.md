# ML

This repo is intended for accelerating ML education and application development.
Refer to the project wiki for setup, resources and hands-on tutorials.

## Introduction

This repository contains foundation classes and utilities for ML applications.
It is under development and subject to change and thus recommended to use this library as a git submodule.

## Installation

1. Install the latest [Miniconda](https://conda.io/en/latest/miniconda.html) on the official website
2. Create a conda environment with python=3.7+
    
    ```
    conda create -n ml37 python=3.7
    ```

3. Restart the terminal to activate the conda env

    ```
    conda activate ml37
    ```

4. Clone the repo and enter the directory

    ```
    git clone --recursive https://gitlab.com/necla-ml/ml.git ML
    cd ML
    ```

5. Add the dependency channels to `~/.condarc`

    ```
    cat recipe/.condarc >> ~/.condarc
    ```

6. Install the dependencies

    ```
    conda install ml
    ```

## Local Development

To utilize GPUs and compile CUDA modules, additional GPU packages are necessary:

- `cudatoolkit` as a dependency of `pytorch` should have been installed
- `cudatoolkit-dev` requires extra space >=16GB for installation

To contribute to this project, follow the development flow:

1. Fork this repo in the beginning
2. Uninstall ML through `conda remove --force ml`
3. Switch to the `dev` branch for development and testing followed by merge back to `main`
    
    ```
    make pull      # Pull submodules recursively
    make dev-setup # Switch to dev branch and build the package for local installation
    git commit ... # Check in modified files
    git push       # Push to the dev branch on the repo
    make merge     # Merge back to the main branch and make a pull request afterwards
    ```

## Conda Distribution

After the merge, one may tag a version and build a conda package for distribution as follows:

```
make tag version=x.y.z
make conda-build
```
