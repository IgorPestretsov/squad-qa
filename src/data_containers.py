from pydantic import BaseModel


class PredictRequestSchema(BaseModel):
    """Request schema for predict endpoint."""

    context: str
    question: str


class PredictResponseSchema(BaseModel):
    """Response schema for predict endpoint."""

    answer: str
    error: str = ''


class UpdateResponseSchema(BaseModel):
    """Response schema for predict endpoint."""

    error: str = ''
