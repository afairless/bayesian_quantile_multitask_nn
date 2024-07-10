
# Handling conda environment in Makefile:
# 	https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# 	Makefile can't use `conda activate`

.ONESHELL:

SHELL = /bin/bash

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

step01:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s01_generate_data.py
	conda deactivate
	
step02:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s02_regress.py
	conda deactivate
	
step03:
	$(CONDA_ACTIVATE) stan
	python src/s03_bayes_stan/bayes_stan.py
	conda deactivate

step03_all: step01 step03
	
step04:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s04_quantile.py
	conda deactivate
	
step05:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s05_singletask_nn.py
	conda deactivate
	
step06:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s06_multitask_nn.py
	
step10:
	$(CONDA_ACTIVATE) distribution_nn02
	python src/s10_results.py
	conda deactivate
