## 2026-02-25 - [Hardcoded Flask Secret Key]
**Vulnerability:** A hardcoded Flask `SECRET_KEY` ('balik2024') was committed directly in the source code.
**Learning:** Developers often hardcode development keys to test Socket.IO or session state and forget to remove them. If an attacker reads the source code, they can sign their own session cookies to impersonate users or bypass auth.
**Prevention:** Always use `os.environ.get()` backed by a secure fallback like `os.urandom(32).hex()` to ensure production deployments use cryptographically secure, unpredictable keys.
