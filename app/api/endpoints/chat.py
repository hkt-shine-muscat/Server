from typing import Any
from fastapi import APIRouter

from app.schema.model import MessageRequest, MessageResponse
from ai_model import use_model

router = APIRouter()


@router.post("", response_model=MessageResponse)
def chat(request: MessageRequest) -> Any:
    chat_histories = ""

    for _, value in enumerate(request.message_history):
        chat_histories += f"{value}</s>"

    message = use_model(chat_histories)

    return MessageResponse(
        message=message
    )
