from typing import List
from pydantic import BaseModel


class PostGet(BaseModel):
    """Model for post representation in web service."""

    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    """Model for representation of web service response."""

    exp_group: str
    recommendations: List[PostGet]
