import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from shl_recommender.domain.schemas import Message
from shl_recommender.core.config import settings
from shl_recommender.core.constants import STATE_EXTRACTOR_SYSTEM_PROMPT, RESPONSE_GENERATOR_SYSTEM_PROMPT


async def _chat_completion_with_fallback(
    gemini_client: AsyncOpenAI,
    groq_client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    timeout: float,
    temperature: float = 0.0,
    json_mode: bool = True
) -> str:
    """
    Tries Gemini primary model first (if API key is configured).
    If it fails, falls back to Groq 70B, then Groq 8B.
    Handles automatic rate-limit (429) retries with backoff sleep.
    """
    clients_and_models = []
    
    gemini_key = gemini_client.api_key if gemini_client else None
    if gemini_key and gemini_key != "dummy":
        clients_and_models.append((gemini_client, settings.GEMINI_LLM_MODEL))
    else:
        print("[LLM] Gemini client not configured or dummy key. Skipping Gemini...")
    
    if groq_client:
        clients_and_models.append((groq_client, settings.GROQ_LLM_MODEL))
        clients_and_models.append((groq_client, settings.GROQ_FALLBACK_MODEL))
        
    if not clients_and_models:
        raise RuntimeError("No LLM clients configured")
        
    for client, model in clients_and_models:
        retries = 3
        backoff = 2.0  # seconds
        
        while retries > 0:
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=timeout
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                # Check for rate limit (429) in error message
                if "429" in error_str or "rate_limit" in error_str.lower() or "limit exceeded" in error_str.lower():
                    retries -= 1
                    if retries > 0:
                        print(f"Rate limit hit for model '{model}'. Retrying in {backoff}s... ({retries} retries left)")
                        await asyncio.sleep(backoff)
                        backoff *= 1.5
                        continue
                
                print(f"LLM call failed with model '{model}': {e}")
                break  # Break inner loop to try next fallback
                
        # If we broke or exhausted retries for the last model, raise
        if (client, model) == clients_and_models[-1]:
            raise RuntimeError("All models failed including fallback")
        print(f"Falling back from '{model}'...")
    
    raise RuntimeError("All models failed")


async def extract_state(
    messages: List[Message],
    turn_limit_reached: bool,
    gemini_client: AsyncOpenAI,
    groq_client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Calls Groq/Gemini Cloud API to extract structured conversation state.
    """
    system_prompt = STATE_EXTRACTOR_SYSTEM_PROMPT
    if turn_limit_reached:
        system_prompt += "\n\nIMPORTANT: Turn limit reached. You MUST set 'recommendations_ready' to true and produce a best-effort recommendation from current context."

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        content = await _chat_completion_with_fallback(
            gemini_client, groq_client, api_messages,
            timeout=settings.STATE_EXTRACTOR_TIMEOUT,
            temperature=0.0,
            json_mode=True
        )
        return json.loads(content)
    except Exception as e:
        print(f"State extractor failed entirely: {e}")
        # Return safe default state
        return {
            "is_vague": False,
            "is_out_of_scope": False,
            "recommendations_ready": True if turn_limit_reached else False,
            "end_of_conversation": False,
            "allowed_test_types": [],
            "semantic_search_term": messages[-1].content if messages else "",
            "items_to_compare": []
        }


async def generate_clarifying_question(
    messages: List[Message],
    gemini_client: AsyncOpenAI,
    groq_client: AsyncOpenAI
) -> str:
    """
    Generates a concise clarifying question.
    """
    clarify_system = (
        "You are a helpful SHL Assessment Recommender assistant. The user's request is too vague. "
        "Ask a concise, polite clarifying question (at most 2 sentences) to narrow down their hiring needs, "
        "such as target role, level of seniority, language requirements, or assessment focus. "
        "Do not output markdown tables or recommendations yet."
    )
    
    api_messages = [{"role": "system", "content": clarify_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        content = await _chat_completion_with_fallback(
            gemini_client, groq_client, api_messages,
            timeout=settings.CLARIFY_GENERATOR_TIMEOUT,
            temperature=0.3,
            json_mode=False
        )
        return content.strip()
    except Exception as e:
        print(f"Clarifying question generation failed entirely: {e}")
        return "Could you please tell me more about the role you are hiring for and what specific skills or behavioral traits you'd like to assess?"


def _minimize_catalog_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    minimized = []
    for item in items:
        desc = item.get("description") or ""
        if len(desc) > 250:
            desc = desc[:250] + "..."
        
        minimized.append({
            "id": item["id"],
            "name": item["name"],
            "test_type": item["test_type"],
            "description": desc,
            "duration": item.get("duration") or "—",
            "languages": item.get("languages", [])[:4]
        })
    return minimized


async def generate_response(
    messages: List[Message],
    retrieved_items: List[Dict[str, Any]],
    previous_ids: List[int],
    gemini_client: AsyncOpenAI,
    groq_client: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Generates the assistant reply and selects recommendations.
    """
    min_items = _minimize_catalog_items(retrieved_items)
    items_json_str = json.dumps(min_items, indent=2, ensure_ascii=False)
    response_gen_system = RESPONSE_GENERATOR_SYSTEM_PROMPT + f"\n\nRetrieved Catalog Items:\n{items_json_str}\n\nPreviously Recommended IDs: {previous_ids}"

    api_messages = [{"role": "system", "content": response_gen_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        content = await _chat_completion_with_fallback(
            gemini_client, groq_client, api_messages,
            timeout=settings.RESPONSE_GENERATOR_TIMEOUT,
            temperature=0.0,
            json_mode=True
        )
        return json.loads(content)
    except Exception as e:
        print(f"Response generator failed entirely: {e}")
        return {
            "assistant_reply": "I have updated the recommended shortlist based on your requirements.",
            "selected_ids": previous_ids if previous_ids else [item["id"] for item in retrieved_items[:3]]
        }
