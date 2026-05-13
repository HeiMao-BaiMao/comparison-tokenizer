import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed - non-OpenAI tokenizers unavailable")

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    logger.warning("sentencepiece not installed")

app = FastAPI(title="Tokenizer Comparison", version="2.0.0")


@dataclass
class TokenizerDef:
    id: str
    name: str
    provider: str
    type: str
    encoding: Optional[str] = None
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    trust_remote_code: bool = False
    available: bool = True
    error_msg: Optional[str] = None


TOKENIZER_DEFS = [
    TokenizerDef(
        id="gpt4o",
        name="GPT-4o / GPT-4.1",
        provider="OpenAI",
        type="tiktoken",
        encoding="o200k_base",
    ),
    TokenizerDef(
        id="qwen35",
        name="Qwen 3.5",
        provider="Alibaba (Qwen)",
        type="transformers",
        model_id="Qwen/Qwen3.5-0.8B",
    ),
    TokenizerDef(
        id="deepseek-v4",
        name="DeepSeek V4",
        provider="DeepSeek AI",
        type="transformers",
        model_id="deepseek-ai/DeepSeek-V4-Flash",
    ),
    TokenizerDef(
        id="glm5",
        name="GLM-5.1",
        provider="Zhipu AI (GLM)",
        type="transformers",
        model_id="zai-org/GLM-5.1",
        trust_remote_code=True,
    ),
    TokenizerDef(
        id="kimi-k2",
        name="Kimi K2.6",
        provider="Moonshot AI (Kimi)",
        type="transformers",
        model_id="moonshotai/Kimi-K2.6",
        trust_remote_code=True,
    ),
    TokenizerDef(
        id="llm-jp-v4",
        name="LLM-jp v4",
        provider="LLM-jp",
        type="sentencepiece",
        model_path="SentencePiece/llm-jp-tokenizer_ver4.0_alpha1.0.model",
    ),
    TokenizerDef(
        id="lfm25",
        name="LFM 2.5",
        provider="Liquid AI",
        type="transformers",
        model_id="LiquidAI/LFM2.5-350M",
    ),
    TokenizerDef(
        id="plamo3",
        name="PLaMo 3",
        provider="Preferred Networks",
        type="transformers",
        model_id="pfnet/plamo-3-nict-2b-base",
    ),
    TokenizerDef(
        id="gemma4",
        name="Gemma 4",
        provider="Google (Gemma)",
        type="transformers",
        model_id="google/gemma-4-E2B-it",
    ),
    TokenizerDef(
        id="minimax-m27",
        name="MiniMax M2.7",
        provider="MiniMax",
        type="transformers",
        model_id="MiniMaxAI/MiniMax-M2.7",
        trust_remote_code=True,
    ),
]

_tokenizer_cache = {}


def get_tokenizer(defn: TokenizerDef):
    if defn.id in _tokenizer_cache:
        return _tokenizer_cache[defn.id]

    try:
        if defn.type == "tiktoken":
            tokenizer = tiktoken.get_encoding(defn.encoding)
            _tokenizer_cache[defn.id] = tokenizer
            return tokenizer
        elif defn.type == "transformers":
            if not HAS_TRANSFORMERS:
                raise ImportError("transformers package not installed")
            tokenizer = AutoTokenizer.from_pretrained(
                defn.model_id,
                trust_remote_code=defn.trust_remote_code,
            )
            _tokenizer_cache[defn.id] = tokenizer
            return tokenizer
        elif defn.type == "sentencepiece":
            if not HAS_SENTENCEPIECE:
                raise ImportError("sentencepiece package not installed")
            model_path = Path(__file__).parent / defn.model_path
            tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
            _tokenizer_cache[defn.id] = tokenizer
            return tokenizer
    except Exception as e:
        defn.available = False
        defn.error_msg = str(e)[:300]
        logger.warning(f"Failed to load {defn.id}: {e}")
        return None


class TokenizeRequest(BaseModel):
    text: str
    tokenizers: list[str] = []


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/tokenizers")
async def list_tokenizers():
    result = []
    for d in TOKENIZER_DEFS:
        tokenizer = get_tokenizer(d)
        result.append({
            "id": d.id,
            "name": d.name,
            "provider": d.provider,
            "type": d.type,
            "available": d.available,
            "error_msg": d.error_msg,
        })
    return {"tokenizers": result}


@app.post("/api/tokenize")
async def tokenize(req: TokenizeRequest):
    text = req.text
    if not text.strip():
        return JSONResponse({"error": "Text cannot be empty"}, status_code=400)

    results = []
    for tid in req.tokenizers:
        defn = next((d for d in TOKENIZER_DEFS if d.id == tid), None)
        if not defn:
            continue

        tokenizer = get_tokenizer(defn)
        if not tokenizer:
            results.append({
                "tokenizer_id": tid,
                "tokenizer_name": defn.name,
                "provider": defn.provider,
                "error": defn.error_msg,
                "available": False,
            })
            continue

        try:
            if defn.type == "tiktoken":
                token_ids = tokenizer.encode(text)
                tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            elif defn.type == "sentencepiece":
                token_ids = tokenizer.encode(text)
                tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
            else:
                encoding = tokenizer.encode(text)
                token_ids = encoding.ids if hasattr(encoding, "ids") else encoding
                tokens = []
                for token_id in token_ids:
                    try:
                        decoded = tokenizer.decode([token_id])
                        tokens.append(decoded if decoded else f"[{token_id}]")
                    except Exception:
                        tokens.append(f"[{token_id}]")

            results.append({
                "tokenizer_id": tid,
                "tokenizer_name": defn.name,
                "provider": defn.provider,
                "token_count": len(token_ids),
                "char_count": len(text),
                "tokens": tokens,
                "token_ids": token_ids,
                "available": True,
            })
        except Exception as e:
            results.append({
                "tokenizer_id": tid,
                "tokenizer_name": defn.name,
                "provider": defn.provider,
                "error": str(e)[:300],
                "available": False,
            })

    results.sort(key=lambda r: r.get("token_count", float("inf")))
    return {"results": results, "char_count": len(text)}


@app.on_event("startup")
async def startup():
    logger.info("Pre-loading tiktoken tokenizers...")
    for defn in TOKENIZER_DEFS:
        if defn.type == "tiktoken":
            get_tokenizer(defn)
    logger.info("Startup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
