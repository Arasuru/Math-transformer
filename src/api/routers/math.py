import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Tuple

from src.api.schemas.math import MathRequest, MathResponse
from src.api.dependencies import get_model, get_tokenizer
from src.api.inference import generate_answer

from src.model.lightning import MathLightningModule
from src.data.tokenizer import CharTokenizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Math operations"])


@router.post("/predict", response_model=MathResponse)
def predict_math(
    request: MathRequest,
    model: MathLightningModule = Depends(get_model),
    tokenizer: CharTokenizer = Depends(get_tokenizer),
):
    logger.info(f"Received prediction request for equation: {request.equation}")

    try:
        # Pass the injected dependencies directly to your inference function
        answer = generate_answer(model, tokenizer, request.equation)
        logger.info(f"Successfully predicted answer: {answer}")

        return MathResponse(
            equation=request.equation, prediction=answer, status="success"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error.")
