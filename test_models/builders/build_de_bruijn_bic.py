"""Write ../model_de_bruijn_bic.py: model_de_bruijn.py with its order R estimated from the data (ORDER_RULE = "bic").
Run after every change of model_de_bruijn.py; tests/test_de_bruijn.py checks that only these three lines differ."""
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
source = (HERE / "model_de_bruijn.py").read_text()
changes = [
    ('NAME = "de_bruijn"\n', 'NAME = "de_bruijn_bic"\n'),
    ('DESCRIPTION = "the de Bruijn hybrid: states read from the last R tokens, started at the merged quotient (R = Markov order)"\n',
     'DESCRIPTION = "the de Bruijn hybrid: states read from the last R tokens, started at the merged quotient (R from the data)"\n'),
    ('ORDER_RULE = "markov"                  # "markov": R = the process\'s Markov order; "bic": estimated from the data\n',
     'ORDER_RULE = "bic"                     # "markov": R = the process\'s Markov order; "bic": estimated from the data\n'),
]
for old, new in changes:
    assert source.count(old) == 1, old
    source = source.replace(old, new)
(HERE / "model_de_bruijn_bic.py").write_text(source)
print("wrote", HERE / "model_de_bruijn_bic.py")
