from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tokenomics_app.api.routes import router as api_router
from tokenomics_app.auth import (
    COOKIE_NAME,
    create_session,
    delete_session,
    verify_credentials,
    verify_session,
)
from tokenomics_app.utils.json_safety import SafeJSONResponse

# ── Paths ──
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ── Routes that don't require authentication ──
_PUBLIC_PATHS = {"/login", "/auth/login"}


# ── Request body for login endpoint ──
class LoginRequest(BaseModel):
    username: str
    password: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Tokenomics Sell Pressure Analyzer",
        default_response_class=SafeJSONResponse,
    )

    # ── Auth middleware ──
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path

        # Always allow public paths and static assets
        if path in _PUBLIC_PATHS or path.startswith("/static/"):
            return await call_next(request)

        token = request.cookies.get(COOKIE_NAME)
        if not verify_session(token):
            # API calls → 401 JSON (so the frontend can handle it)
            if path.startswith("/api/"):
                return JSONResponse(
                    {"detail": "Not authenticated"},
                    status_code=401,
                )
            # All other pages → redirect to login
            next_url = request.url.path
            return RedirectResponse(
                url=f"/login?next={next_url}",
                status_code=302,
            )

        return await call_next(request)

    # ── CORS (kept for local dev convenience) ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    # ── Auth routes ──
    @app.get("/login")
    async def login_page():
        return FileResponse(str(_STATIC_DIR / "login.html"))

    @app.post("/auth/login")
    async def auth_login(data: LoginRequest, response: Response):
        if not verify_credentials(data.username, data.password):
            return JSONResponse(
                {"detail": "Invalid username or password."},
                status_code=401,
            )
        token = create_session()
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            key=COOKIE_NAME,
            value=token,
            httponly=True,          # not accessible from JavaScript
            samesite="lax",         # CSRF protection
            secure=False,           # set True when behind HTTPS in production
            max_age=86_400,         # 24 h
            path="/",
        )
        return resp

    @app.post("/auth/logout")
    async def auth_logout(request: Request):
        token = request.cookies.get(COOKIE_NAME)
        if token:
            delete_session(token)
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(key=COOKIE_NAME, path="/")
        return resp

    # ── API routes ──
    app.include_router(api_router, prefix="/api")

    # ── Static files ──
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ── Main app page (auth-protected via middleware) ──
    @app.get("/")
    async def root():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    return app


app = create_app()


def run():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
