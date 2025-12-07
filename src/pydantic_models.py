from typing import Literal

from pydantic import BaseModel, Field


class Diamond(BaseModel):
    carat: float = Field(ge=0.2, le=5.01)
    cut: Literal["Ideal", "Premium", "Very Good", "Good", "Fair"]
    color: Literal["G", "E", "F", "H", "D", "I", "J"]
    clarity: Literal["SI1", "VS2", "IF", "VVS2", "VVS1", "SI2", "I1", "VS1"]
