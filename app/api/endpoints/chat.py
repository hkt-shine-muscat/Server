from typing import Any, List
from fastapi import APIRouter

from schema.model import Hello

router = APIRouter()

@router.get("/", response_model=Hello)
def hello() -> Any:
    return Hello("Hello!")