# Tokenomics Platform Architecture

## Folder Structure

```
tokenomics-app/
├── server.py
├── static/
│   └── index.html
└── tokenomics_app/
    ├── __init__.py
    ├── main.py
    ├── schemas.py
    ├── api/
    │   ├── __init__.py
    │   └── routes.py
    ├── services/
    │   ├── __init__.py
    │   ├── deterministic.py
    │   └── monte_carlo.py
    └── utils/
        ├── __init__.py
        └── json_safety.py
```

## Responsibility Split

- `server.py`
  - Backward-compatible launch entrypoint (`python3 server.py`).
- `tokenomics_app/main.py`
  - FastAPI app factory, middleware, static mount, root page.
- `tokenomics_app/schemas.py`
  - All request schemas and validation rules.
- `tokenomics_app/api/routes.py`
  - HTTP routes and input-level API checks.
- `tokenomics_app/services/deterministic.py`
  - Deterministic release schedule and sell-pressure table logic.
- `tokenomics_app/services/monte_carlo.py`
  - Monte Carlo simulation engine and risk metrics.
- `tokenomics_app/utils/json_safety.py`
  - JSON-safe response handling and float sanitization.

## Design Principles Used

- Clear separation of concerns.
- Service layer isolated from HTTP layer.
- Schema validation centralized in one place.
- UI kept separate from backend Python package.
- Backward compatibility preserved for existing run commands.
# deployed Sat Feb 28 06:54:41 +07 2026
