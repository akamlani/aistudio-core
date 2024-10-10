PACKAGE_NAME=aistudio
PACKAGE_INSTALL_NAME=aistudio-core

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

#################### To Disable SSL Verification
# conda config --set ssl_verify false

#################### Read Environment
RUNTIME_FILE := ./config/runtimes/python.env
include ${RUNTIME_FILE}
export  $(shell sed 's/=.*//' ${RUNTIME_FILE})


#################### Makefile Configuration
# e.g., Darwin for MacOS
PLATFORM_TYPE = $(shell uname)
# dynamically detect shell type as bash or zsh
ifeq ($(shell basename $(SHELL)), zsh)
        SHELL := zsh
else
        SHELL := bash
endif
# dynamically detect if conda is installed
ifeq ($(shell which conda),)
        HAS_CONDA=False
else
        HAS_CONDA=True
endif

#################### Makefile Context
.DEFAULT_GOAL := info

.PHONY: help info reader
help:
	@echo "Commands  : "
	@echo "download  : downloads a new anaconda distribution"
	@echo "install   : create an anaconda and virtual environment based on project $(PACKAGE_INSTALL_NAME)"
	@echo "setup     : executes download and install based on project $(PACKAGE_INSTALL_NAME)"
	@echo "format    : formatting and linting of project $(PACKAGE_NAME)"
	@echo "clean     : cleans all files or project $(PACKAGE_INSTALL_NAME)"
	@echo "cleanall  : executes clean and removes anaconda, and virtual environments"
	@echo "test      : execute unit testing"

info:
	@echo "Info      : "
	@echo "info      : platform=$(PLATFORM_TYPE), shell=$(SHELL)"
	@echo "Kernels:  : $$(jupyter kernelspec list)"

reader:
	@echo "Conda DWLD URI: ${CONDA_DOWNLOAD_URI}"
	@echo "Conda Install:  ${CONDA_INSTALL_PREFIX}"
	@echo "Conda Binary:   ${CONDA_BIN}"
	@echo "Conda Env Name: ${ENV_NAME}"
	@echo "Conda Env File: ${ENV_FILE}"
	@echo "Python Version: ${PYTHON_VERSION}"
	@echo "Python Venv:    ${PYTHON_VENV_PATH}"
	@echo "Env Dep Path:   ${ENV_DEP_PATH}"

#################### Conda Setup
.PHONY: conda_download conda_env_install conda_env_from_file

conda_download:
	curl $(CONDA_DOWNLOAD_URI) > anaconda.sh
	chmod +x anaconda.sh
	mkdir -p $(CONDA_INSTALL_PREFIX)
	./anaconda.sh -u -p $(CONDA_INSTALL_PREFIX)
	rm ./anaconda.sh
	@echo "Downloaded to: ${CONDA_INSTALL_PREFIX}"

.ONESHELL:
conda_env_install:
	conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) && 			\
	source $$(conda info --base)/etc/profile.d/conda.sh  && 			\
	conda activate $(ENV_NAME) && 										\
	pip install -U pip &&												\
	pip install -U ipykernel ipython
	@echo "Installed ENV: ${ENV_NAME}"

.ONESHELL:
conda_env_from_file:
	conda env create -f $(ENV_FILE)

.ONESHELL:
conda env_remove:
	conda env remove -n $(ENV_NAME) -all
