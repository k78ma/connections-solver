# NYT Connections Solver
This project aims to play the NYT Connections Game using vectorized embeddings of words.

## Requirements
- Gensim (`pip install gensim`)
    - You may need to downgrade `scipy` to a version below `1.13.0` for this to work.
- DEAP (`pip install deap`)
- numpy (`pip install numpy`)

## Usage
- The script is written to run by taking a list of words in a `.txt` file in the `puzzles/` folder.
- Each word in the file is separated by a newline. For an example, see [4-18-2024.txt](puzzles/4-18-2024.txt).
- Then, the script can be run by `python solver.py [filename]`. For example:
```
python solver.py 4-18-2024.txt
```

## TODO
- [] Add evaluation against actual solution