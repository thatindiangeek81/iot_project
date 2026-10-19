"""Lets the tests import the server's modules (they live one folder up)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
