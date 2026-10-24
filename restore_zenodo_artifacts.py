#!/usr/bin/env python3
"""Compatibility entry point for the processed v2.1.0 artifact downloader."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "--download")]
    runpy.run_path(str(Path(__file__).with_name("restore_processed_artifacts.py")), run_name="__main__")
