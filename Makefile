clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	find . -name ".DS_Store" -delete

demo:
	python demo.py

windows:
	pyinstaller -c -F --clean --hidden-import=cv2 --hidden-import=supervision --hidden-import=addict --hidden-import=chex --hidden-import=lap --hidden-import=optax --hidden-import=einshape --hidden-import=haiku --hidden-import=mediapy --name sidecar --specpath dist --distpath dist examples/fastapi-pyinstaller/server.py

mac:
	pyinstaller -c -F --clean --name sidecar --specpath dist --distpath dist examples/fastapi-pyinstaller/server.py