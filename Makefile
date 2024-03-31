clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	find . -name ".DS_Store" -delete

demo:
	python demo.py

build:
	pyinstaller -c -F --clean --name main-x86_64-pc-windows-msvc --distpath dist src/index.py