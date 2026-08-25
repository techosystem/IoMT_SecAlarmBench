.PHONY: help install data benchmark all rq1 rq2 rq3 rq4 leakage figures test clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:   ## install pinned dependencies
	pip install -r requirements.txt

data:      ## download CinC 2015 (WUSTL must be requested separately)
	bash scripts/get_data.sh

benchmark: ## generate the 78,729-window benchmark
	python src/injection_generator.py --config configs/benchmark.yaml --out data/benchmark/

all:       ## full five-seed reproduction, deterministic kernels
	python scripts/run_all.py --config configs/benchmark.yaml --seeds 0 1 2 3 4 --deterministic

rq1:       ## Table III -- detection
	python src/rq1_report.py
rq2:       ## Table IV -- threshold-free + matched-budget
	python src/eval_harness.py --matched-far 0.10
rq3:       ## Table V -- 3-way discrimination
	python src/rq3_discrimination.py
rq4:       ## efficiency profile
	python src/eval_harness.py --profile
leakage:   ## WUSTL identifier-leakage audit
	python src/wustl_leakage.py

figures:   ## regenerate all paper figures
	python figures/make_fig_lift.py --width single
	python figures/make_fig2.py
	python figures/make_fig3_stealthiness.py

test:      ## smoke tests
	pytest -q tests/

clean:     ## remove caches (leaves results/ and data/ alone)
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
