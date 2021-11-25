from typing import Any
from fastapi import APIRouter

from app.schema.model import MessageRequest, MessageResponse

router = APIRouter()


@router.post("", response_model=MessageResponse)
def chat(request: MessageRequest) -> Any:
    chat_histories = ""

    for _, value in enumerate(request.message_history):
        chat_histories += f"{value}</s>"

    return MessageResponse(
        message=chat_histories
    )
