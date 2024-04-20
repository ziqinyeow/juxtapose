clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	find . -name ".DS_Store" -delete

demo:
	python demo.py

b:
	pyinstaller -c -F --clean --name sidecar --specpath dist --distpath dist src/server.py