from typing import Literal
from pydantic import BaseModel, Field

class Media(BaseModel):
    user_id: str = Field(default="")
    media_type: Literal["footage_regular", "footage_opening", "image", "bgm"]
    media_id: str = Field(default="")
    media_path:str = Field(default="")

class DataPathRequest(BaseModel):
    data_path: str = Field(..., description="data path for all orgs")