from fastapi import Request
from app.fastapi.api.state import AppState


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
