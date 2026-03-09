# Sentinel's Journal - Balik_Projesi Security Learnings

## 2025-12-13 - Hardcoded SSH Password in Deployment Script
**Vulnerability:** SSH password `'1356'` was hardcoded in `deploy_to_pi.bat` line 57-58, visible in plain text.
**Learning:** Deployment scripts often contain "convenience" instructions that inadvertently expose secrets. The password was added as a user-friendly hint but ended up in version control.
**Prevention:** 
- Never include passwords in deployment scripts, even as "hints"
- Use SSH key-based authentication instead of passwords
- Add pre-commit hooks to scan for common password patterns
- Review deployment scripts with same rigor as application code

---
