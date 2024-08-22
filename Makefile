.PHONY: help clean install pre-commit venv

SHELL := /bin/bash

PROJECT_NAME := $(shell basename -s .git `git config --get remote.origin.url`)
PWD := $(shell pwd)
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRIPT_DIR := scripts

# For pre-commit
export PRE_COMMIT_HOME=.pre-commit

# For poetry
export POETRY_VIRTUALENVS_IN_PROJECT=true
export POETRY_VIRTUALENVS_CREATE=false

# For more information on this technique, see
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

help: ## Show this help message
	@echo -e "\nUsage: make TARGET\n\nTargets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

clean: ## Cleanup
	@rm -rf .pre-commit
	@rm -rf .venv

install: ## Install dependencies
	@poetry install

pre-commit: ## Run pre-commit hooks
	@poetry run pre-commit run --all-files

venv: ## Activate the virtual environment
	@poetry shell
