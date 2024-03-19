# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities via GitHub Security Advisories (not public issues).

**Do not** include sensitive information in public issues.

We will acknowledge within 48 hours and patch critical issues within 7 days.

## Security Notes

- nano-finbert does not handle user credentials or financial data in production
- The API server is for inference only — no user data is stored
- Model weights are local and not transmitted externally
- Do not expose the inference API to the public internet without authentication
