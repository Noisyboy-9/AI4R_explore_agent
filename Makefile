SHELL := /bin/bash

SCRIPT_DIR := utils/scripts/second_run
CONDA_ENV_DIR := Exploring_agent_DRL
CONDA_ENV_FILE := environment.yml
CONDA_ENV_NAME := aiar-rl-explore-gpu

.PHONY: setup ahmet matthijs balaji nitin

setup:
	if conda env list | awk '{print $$1}' | grep -Fxq '$(CONDA_ENV_NAME)'; then \
		cd $(CONDA_ENV_DIR) && conda env update --name $(CONDA_ENV_NAME) --file $(CONDA_ENV_FILE) --prune; \
	else \
		cd $(CONDA_ENV_DIR) && conda env create --name $(CONDA_ENV_NAME) --file $(CONDA_ENV_FILE); \
	fi
	conda install -n $(CONDA_ENV_NAME) -y -c conda-forge urllib3 requests

ahmet: setup
	conda run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_ahmet.sh

matthijs: setup
	conda run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_matthijs.sh

balaji: setup
	conda run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_balaji.sh

nitin: setup
	conda run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_nitin.sh
