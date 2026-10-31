# Developer shortcuts for the Build-Failure Prediction project.
.PHONY: help install data train evaluate test api docker-build docker-run clean

help:
	@echo "install       install Python dependencies"
	@echo "data          generate the synthetic dataset"
	@echo "train         train and save the best model"
	@echo "evaluate      evaluate the saved model"
	@echo "test          run the pytest suite"
	@echo "api           start the Flask API on :5000"
	@echo "docker-build  build the Docker image"
	@echo "docker-run    run the Docker image on :5000"
	@echo "clean         remove generated artefacts"

install:
	python -m pip install -r requirements.txt

data:
	python main.py generate-data

train:
	python main.py train

evaluate:
	python main.py evaluate

test:
	python -m pytest -v

api:
	python -m build_failure.api

docker-build:
	docker build -t build-failure-prediction:latest .

docker-run:
	docker run --rm -p 5000:5000 build-failure-prediction:latest

clean:
	rm -rf data/raw data/processed models reports .pytest_cache .mypy_cache
