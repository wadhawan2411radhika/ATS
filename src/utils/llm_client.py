"""
LLM client abstraction.
Supports OpenAI and Groq (same API shape).
Handles structured JSON output with Pydantic schema validation.

OpenAI path: uses client.beta.chat.completions.parse() — native Structured Outputs.
  The model receives the exact Pydantic schema and is constrained to match it.
  Zero chance of the model echoing the schema back as values.

Groq path: JSON mode + prompt with a flat field list (not raw JSON Schema).
  model_json_schema() produces $defs/$ref which confuses models into returning
  the schema definition instead of filling it. We generate a human-readable
  field description instead.
"""

import os
import json
import time
import random
import logging
from typing import Type, TypeVar

from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _backoff(attempt: int, base: float = 2.0, cap: float = 30.0) -> None:
    """Exponential backoff with jitter. Called on 429 or transient errors."""
    delay = min(cap, base ** attempt) + random.uniform(0, 1)
    logger.warning(f"Rate limited — waiting {delay:.1f}s before retry (attempt {attempt})")
    time.sleep(delay)


def _get_client() -> OpenAI:
    provider = config.model.llm_provider

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        return OpenAI(api_key=api_key)

    elif provider == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _flat_field_description(schema: Type[BaseModel]) -> str:
    """
    Build a human-readable field description for Groq prompts.

    Avoids dumping raw JSON Schema (with $defs/$ref) which causes models to
    echo the schema structure back instead of extracting values.

    Produces output like:
        Fields to return (JSON object):
        - required_skills (array of strings): Atomic skill tokens...
        - required_years_of_experience (number or null): Minimum years...
    """
    raw = schema.model_json_schema()
    props = raw.get("properties", {})
    defs = raw.get("$defs", {})
    lines = ["Fields to return as a JSON object:"]

    for field_name, field_info in props.items():
        # Resolve $ref if present
        if "$ref" in field_info:
            ref_key = field_info["$ref"].split("/")[-1]
            field_info = defs.get(ref_key, field_info)

        ftype = field_info.get("type", "")
        if not ftype:
            # anyOf / array of types
            any_of = field_info.get("anyOf", [])
            types = [t.get("type", "") for t in any_of if "type" in t]
            ftype = " or ".join(t for t in types if t) or "any"

        desc = field_info.get("description", "")
        # Truncate long descriptions
        if len(desc) > 120:
            desc = desc[:117] + "..."

        lines.append(f"- {field_name} ({ftype}): {desc}")

    return "\n".join(lines)


def extract_structured(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    retries: int = 2,
) -> T:
    """
    Call LLM and parse response into a Pydantic model.

    OpenAI: uses beta.chat.completions.parse() — native Structured Outputs.
    Groq: JSON mode with flat field description prompt.
    """
    provider = config.model.llm_provider

    if provider == "openai":
        return _extract_openai(system_prompt, user_prompt, schema, retries)
    else:
        return _extract_groq(system_prompt, user_prompt, schema, retries)


def _extract_openai(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    retries: int,
) -> T:
    """
    OpenAI Structured Outputs via client.beta.chat.completions.parse().
    Handles 429 rate limits with exponential backoff + jitter.
    """
    client = _get_client()

    for attempt in range(retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=config.model.llm_model,
                temperature=config.model.llm_temperature,
                max_tokens=config.model.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema,
            )
            result = response.choices[0].message.parsed
            if result is None:
                raise ValueError("Model returned null — possible refusal.")
            return result

        except RateLimitError:
            if attempt < retries:
                _backoff(attempt + 1)
            else:
                raise RuntimeError("Rate limit exceeded after all retries.")

        except Exception as e:
            logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1.5 ** attempt)
            else:
                raise RuntimeError(
                    f"Failed to extract structured data after {retries + 1} attempts. "
                    f"Last error: {e}"
                )


def _extract_groq(
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    retries: int,
) -> T:
    """
    Groq path: JSON mode with a flat human-readable field description.
    Avoids raw JSON Schema which causes Groq models to echo schema structure.
    """
    client = _get_client()

    field_desc = _flat_field_description(schema)
    full_system = (
        f"{system_prompt}\n\n"
        f"{field_desc}\n\n"
        f"Return ONLY a valid JSON object. No markdown fences, no explanation, no preamble."
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model.llm_model,
                temperature=config.model.llm_temperature,
                max_tokens=config.model.llm_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw = response.choices[0].message.content
            # Strip accidental markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = schema.model_validate_json(raw)
            return parsed

        except Exception as e:
            logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1.5 ** attempt)
            else:
                raise RuntimeError(
                    f"Failed to extract structured data after {retries + 1} attempts. "
                    f"Last error: {e}"
                )