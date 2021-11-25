from typing import List
from pydantic import BaseModel

class MessageRequest(BaseModel):
    message_history: List[str]
    message: str

class MessageResponse(BaseModel):
    message: str