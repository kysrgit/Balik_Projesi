# Security Policy

## Reporting a Vulnerability

This project is a private research prototype for the "Underwater Pufferfish Detection System".

**Do not report security vulnerabilities publicly.**

If you discover a security vulnerability, please report it directly to the project owner via private communication channels.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Best Practices for Contributors

1.  **Never commit secrets:** API keys, passwords, and tokens should never be pushed to the repository. Use `.env` files (which are git-ignored).
2.  **Review dependencies:** Before adding new libraries, check them for known vulnerabilities.
3.  **Private Data:** Do not upload datasets containing sensitive personal information (PII) to this repository.
