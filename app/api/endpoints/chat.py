from typing import Any
from fastapi import APIRouter

from app.schema.model import MessageRequest, MessageResponse

router = APIRouter()

@router.post("", response_model=MessageResponse)
def chat(request: MessageRequest) -> Any:

    return MessageResponse(
        message=request.message
    )