SHELL := /bin/bash

SCRIPT_DIR := utils/scripts/second-pass
CONDA_ENV_DIR := Exploring_agent_DRL
CONDA_ENV_FILE := environment.yml
CONDA_ENV_NAME := aiar-rl-explore-gpu
CONDA_BIN := /opt/miniconda3/bin/conda

.PHONY: setup ahmet matthijs balaji nitin

setup:
	if $(CONDA_BIN) env list | awk '{print $$1}' | grep -Fxq '$(CONDA_ENV_NAME)'; then \
		cd $(CONDA_ENV_DIR) && $(CONDA_BIN) env update --name $(CONDA_ENV_NAME) --file $(CONDA_ENV_FILE) --prune; \
	else \
		cd $(CONDA_ENV_DIR) && $(CONDA_BIN) env create --name $(CONDA_ENV_NAME) --file $(CONDA_ENV_FILE); \
	fi
	$(CONDA_BIN) install -n $(CONDA_ENV_NAME) -y -c conda-forge urllib3 requests "numpy<2"

ahmet: setup
	$(CONDA_BIN) run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_ahmet.sh

matthijs: setup
	$(CONDA_BIN) run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_matthijs.sh

balaji: setup
	$(CONDA_BIN) run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_balaji.sh

nitin: setup
	$(CONDA_BIN) run -n $(CONDA_ENV_NAME) bash $(SCRIPT_DIR)/run_exploring_agent_cpu_nitin.sh
