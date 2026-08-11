from pydantic import BaseModel, Field

class MathRequest(BaseModel):
    equation: str = Field(..., example="124+56=", description="The mathematical equation to solve.")

class MathResponse(BaseModel):
    equation: str
    prediction: str
    status: str