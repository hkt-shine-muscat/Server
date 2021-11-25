from typing import List
from enum import Enum
from pydantic import BaseModel


class Type(str, Enum):
    user = "user"
    bot = "bot"


class Message(BaseModel):
    message: str
    type: Type

    class Config:
        use_enum_values = True


class MessageRequest(BaseModel):
    message_history: List[Message]
    message: str


class MessageResponse(BaseModel):
    message: str
