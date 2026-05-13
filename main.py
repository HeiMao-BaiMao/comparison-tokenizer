import asyncio
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import tiktoken

load_dotenv()

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

app = FastAPI(title="Tokenizer Comparison", version="3.0.0")

MAX_CONCURRENT_LOADS = 3
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30

_tokenizer_cache = {}
_load_locks: dict[str, asyncio.Lock] = {}
_load_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LOADS)
_session_requests: dict[str, list[float]] = {}


def _cleanup_rate_limits():
    now = time.time()
    expired = [sid for sid, times in _session_requests.items()
               if not times or now - times[-1] > RATE_LIMIT_WINDOW * 2]
    for sid in expired:
        del _session_requests[sid]


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
    hf_token_env: Optional[str] = None
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
        trust_remote_code=True,
        hf_token_env="plamo_token",
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


async def get_tokenizer(defn: TokenizerDef):
    if defn.id in _tokenizer_cache:
        return _tokenizer_cache[defn.id]

    if defn.id not in _load_locks:
        _load_locks[defn.id] = asyncio.Lock()

    async with _load_locks[defn.id]:
        if defn.id in _tokenizer_cache:
            return _tokenizer_cache[defn.id]

        try:
            if defn.type == "tiktoken":
                tokenizer = await asyncio.to_thread(
                    tiktoken.get_encoding, defn.encoding
                )
            elif defn.type == "transformers":
                if not HAS_TRANSFORMERS:
                    raise ImportError("transformers package not installed")
                kwargs = {"trust_remote_code": defn.trust_remote_code}
                if defn.hf_token_env:
                    token = os.getenv(defn.hf_token_env)
                    if token:
                        kwargs["token"] = token
                    else:
                        raise ValueError(
                            f"Environment variable '{defn.hf_token_env}' not set."
                        )
                async with _load_semaphore:
                    tokenizer = await asyncio.to_thread(
                        AutoTokenizer.from_pretrained, defn.model_id, **kwargs
                    )
            elif defn.type == "sentencepiece":
                if not HAS_SENTENCEPIECE:
                    raise ImportError("sentencepiece package not installed")
                model_path = Path(__file__).parent / defn.model_path

                def _load_sp():
                    return spm.SentencePieceProcessor(model_file=str(model_path))

                tokenizer = await asyncio.to_thread(_load_sp)
            else:
                raise ValueError(f"Unknown tokenizer type: {defn.type}")

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
    session_id: str = ""


def _check_rate_limit(session_id: str) -> bool:
    if not session_id:
        return True
    now = time.time()
    times = _session_requests.get(session_id, [])
    times = [t for t in times if now - t < RATE_LIMIT_WINDOW]
    if len(times) >= RATE_LIMIT_MAX:
        return False
    times.append(now)
    _session_requests[session_id] = times
    if len(_session_requests) > 1000:
        _cleanup_rate_limits()
    return True


def _run_tiktoken(tokenizer, text: str):
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return token_ids, tokens


def _run_sentencepiece(tokenizer, text: str):
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    return token_ids, tokens


def _run_transformers(tokenizer, text: str):
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids if hasattr(encoding, "ids") else encoding
    tokens = []
    for token_id in token_ids:
        try:
            decoded = tokenizer.decode([token_id])
            tokens.append(decoded if decoded else f"[{token_id}]")
        except Exception:
            tokens.append(f"[{token_id}]")
    return token_ids, tokens


async def _tokenize_one(defn: TokenizerDef, text: str):
    tokenizer = await get_tokenizer(defn)
    if not tokenizer:
        return {
            "tokenizer_id": defn.id,
            "tokenizer_name": defn.name,
            "provider": defn.provider,
            "error": defn.error_msg,
            "available": False,
        }

    try:
        if defn.type == "tiktoken":
            token_ids, tokens = await asyncio.to_thread(
                _run_tiktoken, tokenizer, text
            )
        elif defn.type == "sentencepiece":
            token_ids, tokens = await asyncio.to_thread(
                _run_sentencepiece, tokenizer, text
            )
        else:
            token_ids, tokens = await asyncio.to_thread(
                _run_transformers, tokenizer, text
            )

        return {
            "tokenizer_id": defn.id,
            "tokenizer_name": defn.name,
            "provider": defn.provider,
            "token_count": len(token_ids),
            "char_count": len(text),
            "tokens": tokens,
            "token_ids": token_ids,
            "available": True,
        }
    except Exception as e:
        return {
            "tokenizer_id": defn.id,
            "tokenizer_name": defn.name,
            "provider": defn.provider,
            "error": str(e)[:300],
            "available": False,
        }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/tokenizers")
async def list_tokenizers():
    result = []
    for d in TOKENIZER_DEFS:
        tokenizer = await get_tokenizer(d)
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

    if not _check_rate_limit(req.session_id):
        return JSONResponse(
            {"error": f"Rate limit exceeded ({RATE_LIMIT_MAX} req/{RATE_LIMIT_WINDOW}s)"},
            status_code=429,
        )

    defns = [d for d in TOKENIZER_DEFS if d.id in req.tokenizers]
    if not defns:
        return JSONResponse({"error": "No valid tokenizers selected"}, status_code=400)

    tasks = [_tokenize_one(d, text) for d in defns]
    results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r.get("token_count", float("inf")))
    logger.info(f"[{req.session_id[:8]}] {len(defns)} tokenizers, {len(text)} chars")
    return {"results": results, "char_count": len(text)}


@app.on_event("startup")
async def startup():
    logger.info("Pre-loading tiktoken tokenizers...")
    for defn in TOKENIZER_DEFS:
        if defn.type == "tiktoken":
            await get_tokenizer(defn)
    logger.info("Startup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
