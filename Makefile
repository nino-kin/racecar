.PHONY: help pre-commit

SHELL := /bin/bash

PROJECT_NAME := $(shell basename -s .git `git config --get remote.origin.url`)
PWD := $(shell pwd)
ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SCRIPT_DIR := scripts

export PRE_COMMIT_HOME=.pre-commit

# For more information on this technique, see
# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

help: ## Show this help message
	@echo -e "\nUsage: make TARGET\n\nTargets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

pre-commit: ## Run pre-commit hooks
	@pre-commit run --all-files
