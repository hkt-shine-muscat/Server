from typing import Any
from fastapi import APIRouter

from app.schema.model import Message

router = APIRouter()

@router.post("/", response_model=Message)
def chat() -> Any:
    return Message(
        message="Test Message"
    )