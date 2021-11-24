from fastapi import FastAPI
import fastapi

def create_app() -> FastAPI:
    fast_api = FastAPI(__name__)
    return fastapi

