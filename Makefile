# Python 3.11: coremltools has no wheels for 3.15 yet. Override with
#   make setup PYTHON=/path/to/python3.11
PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

MODEL_ID ?= vitouphy/wav2vec2-xls-r-300m-timit-phoneme
OUT      ?= models
RUNS     ?= 10

.PHONY: help setup convert bench all clean

help:
	@echo "make setup     create .venv and install dependencies"
	@echo "make convert   build FP32, FP16 and INT8 packages into $(OUT)/"
	@echo "make bench     measure size, load, latency, agreement -> benchmarks/results.md"
	@echo "make all       convert, then bench"
	@echo "make clean     remove $(OUT)/"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

convert:
	$(PY) src/convert.py --model-id $(MODEL_ID) --out $(OUT) --precision all

bench:
	$(PY) src/benchmark.py --models $(OUT) --runs $(RUNS) --out benchmarks/results.md

all: convert bench

clean:
	rm -rf $(OUT)
