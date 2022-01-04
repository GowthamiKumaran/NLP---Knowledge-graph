from pydantic import BaseModel
from typing import Optional


class LongText(BaseModel):
    text : Optional[str]
    description: Optional[str]
    text_type: Optional[str]

class word(BaseModel):
    description: Optional[str]
    url: Optional[str]
