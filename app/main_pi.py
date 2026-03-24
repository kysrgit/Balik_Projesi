#!/usr/bin/env python3
"""
Deprecated Raspberry Pi entrypoint.

This wrapper is kept only for backwards compatibility and delegates to the
shared runtime in app.main so Pi and PC execution paths stay aligned.
"""

from app.main import main as shared_main


def main():
    print("UYARI: app/main_pi.py deprecated. Ortak runtime olan app/main.py calistiriliyor.")
    shared_main()


if __name__ == "__main__":
    main()
