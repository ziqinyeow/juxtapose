clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	find . -name ".DS_Store" -delete

demo:
	python demo.py