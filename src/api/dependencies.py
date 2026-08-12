from fastapi import Request, HTTPException
import logging

logger = logging.getLogger(__name__)


def get_model(request: Request):
    model = request.app.state.model
    if not model:
        logger.error("Attempted to access the model before it was loaded,")
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    return model


def get_tokenizer(request: Request):
    tokenizer = request.app.state.tokenizer
    if not tokenizer:
        logger.error("Attempted to access the tokenizer before it was loaded.")
        raise HTTPException(
            status_code=503, detail="Tokenizer is currently unavailable"
        )

    return tokenizer
