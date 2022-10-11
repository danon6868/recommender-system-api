from pydantic import BaseModel


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
