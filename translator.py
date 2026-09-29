"""OpenAI ChatGPT 모델을 사용한 번역 로직."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AuthenticationError, OpenAI, OpenAIError, RateLimitError

load_dotenv(Path(__file__).parent / ".env")

DEFAULT_MODEL = "gpt-4o-mini"
MAX_CHARS = 2000
LANGUAGES = {
    "en": {"name": "English", "label": "영어"},
    "ja": {"name": "日本語", "label": "일본어"},
    "vi": {"name": "Tiếng Việt", "label": "베트남어"},
}
TONES = {
    "자연스럽게": "Use a natural tone that matches the original text.",
    "격식체": "Use a formal, polite business tone.",
    "친근하게": "Use a casual, friendly conversational tone.",
}
_PROMPT_NAMES = {"en": "English", "ja": "Japanese", "vi": "Vietnamese"}


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 번역 오류."""


def get_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def get_model() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def build_system_prompt(targets: list[str], tone: str) -> str:
    names = ", ".join(_PROMPT_NAMES[code] for code in targets)
    keys = ", ".join(f'"{code}"' for code in targets)
    return (
        "You are a professional translator.\n"
        f"Translate the user's text into {names}.\n"
        "Detect the source language automatically.\n"
        "Keep the original meaning and formatting. Do not add explanations.\n"
        f"{TONES[tone]}\n"
        f"Respond ONLY in JSON with keys: {keys}."
    )


def translate(client: OpenAI, model: str, text: str, targets: list[str], tone: str = "자연스럽게") -> dict[str, str]:
    if not text.strip():
        raise TranslationError("번역할 텍스트를 입력하세요.")
    if len(text) > MAX_CHARS:
        raise TranslationError(f"텍스트는 최대 {MAX_CHARS:,}자까지 번역할 수 있습니다.")
    if not targets:
        raise TranslationError("번역할 언어를 하나 이상 선택하세요.")

    messages = [
        {"role": "system", "content": build_system_prompt(targets, tone)},
        {"role": "user", "content": text},
    ]
    # JSON 파싱에 실패하면 1회 재시도
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=messages,
            )
            result = json.loads(response.choices[0].message.content or "")
            return {code: str(result.get(code, "")).strip() for code in targets}
        except json.JSONDecodeError:
            if attempt == 1:
                raise TranslationError("번역 결과를 해석하지 못했습니다. 다시 시도하세요.")
        except AuthenticationError:
            raise TranslationError("API 키가 유효하지 않습니다. `.env`의 OPENAI_API_KEY를 확인하세요.")
        except RateLimitError:
            raise TranslationError("요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도하세요.")
        except APITimeoutError:
            raise TranslationError("응답 시간이 초과되었습니다. 다시 시도하세요.")
        except APIConnectionError:
            raise TranslationError("네트워크 오류가 발생했습니다. 인터넷 연결을 확인하세요.")
        except OpenAIError as e:
            raise TranslationError(f"번역 중 오류가 발생했습니다: {e}")
