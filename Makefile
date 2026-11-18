.PHONY: help install data paper all rq1 rq3 leakage figures test clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

install:   ## install pinned dependencies
	pip install -r requirements.txt

data:      ## download CinC 2015 (WUSTL must be requested separately)
	bash scripts/get_data.sh

paper:     ## reproduce the paper: all four attacker settings, every analysis
	python scripts/reproduce_paper.py

all:       ## single-setting (C1-PPG) pipeline, five seeds, deterministic
	python scripts/run_all.py --stages config,checks,table2,rq1,matched,multiseed,rq3,wustl,summary --seeds 0,1,2,3,4

rq1:       ## C1-PPG detection report only
	python scripts/run_all.py --stages rq1
rq3:       ## 3-way and 4-way discrimination only
	python scripts/run_all.py --stages rq3
leakage:   ## WUSTL identifier-leakage audit only
	python scripts/run_all.py --stages wustl

figures:   ## Fig. 2 from results/paper
	python figures/make_fig2_grid.py

test:      ## smoke tests
	pytest -q tests/

clean:     ## remove caches (leaves results/ and data/ alone)
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
