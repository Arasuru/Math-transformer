import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path
import uvicorn

from src.data.tokenizer import CharTokenizer
from src.model.lightning import MathLightningModule
from src.api.routers import math


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("Initializing application and loading model checkpoints...")

    checkpoints_dir = Path("checkpoints")
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))

    if not ckpt_files:
        logger.error("Startup failed: No checkpoints found.")
        raise RuntimeError("No checkpoints found in /checkpoints folder.")

    ckpt_path = ckpt_files[0]
    tokenizer = CharTokenizer()

    # Load model
    model = MathLightningModule.load_from_checkpoint(
        ckpt_path, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id
    )
    model.eval()

    # Inject into application state
    app.state.model = model
    app.state.tokenizer = tokenizer

    logger.info(f"✅ Production Model Loaded: {ckpt_path.name}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down and clearing memory...")
    app.state.model = None
    app.state.tokenizer = None


app = FastAPI(
    title="Math Transformer API",
    description="A production-grade AI that does math character-by-character.",
    lifespan=lifespan,
)

app.include_router(math.router)

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
