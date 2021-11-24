from fastapi import APIRouter
from .endpoints import chat

api_router = APIRouter()
api_router.include_router(, prefix="/chat", tags=["채팅"])