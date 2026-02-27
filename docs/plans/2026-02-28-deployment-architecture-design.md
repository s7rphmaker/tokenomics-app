# Deployment Architecture Design
**Date:** 2026-02-28

## Overview
Multi-site deployment architecture with staging/production environments, rollback capability, and security hardening. Supports up to 5 sites on a single VPS.

## Architecture: GitHub + Deploy Script (Variant A)

### Server Folder Structure
```
/var/www/
├── tokenomics/
│   ├── production/     ← https://tokenomicsbuilder.xyz
│   └── staging/        ← https://staging.tokenomicsbuilder.xyz
├── site2/
│   ├── production/
│   └── staging/
└── ...
```

### Nginx
- `/etc/nginx/sites-available/tokenomics-production`
- `/etc/nginx/sites-available/tokenomics-staging`
- One config per site per environment

### Systemd Services
- `tokenomics-production.service`
- `tokenomics-staging.service`
- One service per site per environment

## Workflow
1. AI agent edits code locally on Mac
2. `git push` → GitHub private repo
3. `deploy staging tokenomics` → test on staging subdomain
4. `deploy production tokenomics` → go live
5. `rollback production tokenomics` → instant rollback if needed

## Deploy Script (`/usr/local/bin/deploy`)
```
deploy <environment> <site>     # pull latest from GitHub and restart
rollback <environment> <site>   # revert to previous commit and restart
```

## Security
- UFW firewall: ports 22, 80, 443 only
- Fail2ban: brute-force protection on SSH
- SSH key-only authentication (password login disabled)
- Unattended security upgrades

## GitHub
- One private repo per site
- Deploy key (read-only SSH key) on server per repo
- Main branch = production, staging branch = staging
