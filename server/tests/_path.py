"""Lets the tests import the server's modules (they live one folder up)."""

import os
import pathlib
import sys

# A Groq key in this laptop's .env would turn the Gemini rate-limit tests into real fallback calls.
os.environ["GROQ_API_KEY"] = ""

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
