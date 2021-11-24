from typing import Any
from fastapi import APIRouter

from app.schema.model import Hello

router = APIRouter()

@router.get("/", response_model=Hello)
def hello() -> Any:
    return Hello("Hello!")