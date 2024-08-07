
# Handling conda environment in Makefile:
# 	https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Makefile can't use `conda activate`

.ONESHELL:

SHELL = /bin/bash

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

step01:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s01_generate_data.py
	conda deactivate
	
step02:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s02_regress.py
	conda deactivate
	
step03:
	$(CONDA_ACTIVATE) stan
	python src/s03_bayes_stan/bayes_stan.py
	conda deactivate

step03_all: step01 step03
	
step04:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s04_quantile.py
	conda deactivate
	
step05:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s05_singletask_nn.py
	conda deactivate
	
step06:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s06_multitask_nn.py
	
step10:
	$(CONDA_ACTIVATE) distribution_nn01
	python src/s10_results.py
	conda deactivate
	
notebook:
	python -c "from src.common import copy_image_files_to_notebook; copy_image_files_to_notebook()"
	$(CONDA_ACTIVATE) jupyter_data_processing02
	jupytext --to notebook notebooks/*.py
	jupyter nbconvert --to markdown notebooks/*.ipynb
	jupyter nbconvert --to html notebooks/*.ipynb
	conda deactivate
	
readme:
	cp notebooks/summary.md README.md
	sed -i 's/!\[image\](\.\/output/!\[image\](\.\/notebooks\/output/g' README.md
