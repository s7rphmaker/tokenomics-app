#!/usr/bin/env python3
"""
Compatibility entrypoint for local runs.

The application is now modularized under `tokenomics_app/`.
Use `python3 server.py` as before.
"""

from tokenomics_app.main import app, run


if __name__ == "__main__":
    run()
