.PHONY: download generate train evaluate demo test clean all

download:
	python scripts/download_data.py

generate:
	python scripts/generate_synthetic.py --num-images 1000

train:
	python scripts/train_detector.py --experiment both

evaluate:
	python scripts/evaluate.py

demo:
	python scripts/launch_demo.py

test:
	pytest tests/ -v

all: download generate train evaluate

clean:
	rm -rf data/ outputs/
