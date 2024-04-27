clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	find . -name ".DS_Store" -delete

demo:
	python demo.py

b:
	pyinstaller -c -F --clean --name sidecar --specpath dist --distpath dist src/server.py
	# pyinstaller -c -F --clean --name sidecar --specpath dist --add-data ../src/Grammar.txt:yapf_third_party/_ylib2to3/Grammar.txt --add-data ../src/PatternGrammar.txt:yapf_third_party/_ylib2to3/PatternGrammar.txt --distpath dist src/server.py