#!/usr/bin/env python3
"""
KV4P-Desktop — Desktop app for the KV4P-HT ham radio board.

Usage:
    ./run.py
    # or
    python run.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kv4p_ht.main import main

if __name__ == "__main__":
    sys.exit(main())
