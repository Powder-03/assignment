import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from shl_recommender.domain.schemas import Message
from shl_recommender.core.config import settings
from shl_recommender.core.constants import STATE_EXTRACTOR_SYSTEM_PROMPT, RESPONSE_GENERATOR_SYSTEM_PROMPT

# Round-robin counter to spread RPM load across Groq keys
_groq_rr_index = 0


async def _chat_completion_with_fallback(
    gemini_client: AsyncOpenAI,
    groq_clients: List[AsyncOpenAI],
    messages: List[Dict[str, str]],
    timeout: float,
    temperature: float = 0.0,
    json_mode: bool = True
) -> str:
    """
    Tries Groq (llama-3.3-70b-versatile) rotating through keys first.
    If all Groq keys hit rate limit, falls back to Qwen, then Gemini.
    Uses round-robin to spread RPM load evenly across keys.
    """
    global _groq_rr_index
    clients_and_models = []
    
    # Round-robin: rotate clients so each request starts from a different key
    if groq_clients:
        n = len(groq_clients)
        start = _groq_rr_index % n
        _groq_rr_index += 1
        rotated = groq_clients[start:] + groq_clients[:start]
    else:
        rotated = []

    # Primary: All Groq clients (llama-3.3-70b-versatile) in rotated order
    for client in rotated:
        clients_and_models.append((client, settings.GROQ_LLM_MODEL))

    # Secondary: All Groq clients with fallback model (qwen3-32b) in rotated order
    for client in rotated:
        clients_and_models.append((client, settings.GROQ_FALLBACK_MODEL))
            
    # Tertiary: Gemini
    gemini_key = gemini_client.api_key if gemini_client else None
    if gemini_key and gemini_key != "dummy":
        clients_and_models.append((gemini_client, settings.GEMINI_LLM_MODEL))
        
    if not clients_and_models:
        raise RuntimeError("No LLM clients configured")
        
    for client, model in clients_and_models:
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
            if "429" in error_str or "rate_limit" in error_str.lower() or "limit exceeded" in error_str.lower():
                print(f"Rate limit hit for model '{model}'. Switching to next key/model...")
            else:
                print(f"LLM call failed with model '{model}': {e}")
            # Continue to next model/client in the list
        # If we broke or exhausted retries for the last model, raise
        if (client, model) == clients_and_models[-1]:
            raise RuntimeError("All models failed including fallback")
        print(f"Falling back from '{model}'...")
    
    raise RuntimeError("All models failed")


async def extract_state(
    messages: List[Message],
    turn_limit_reached: bool,
    gemini_client: AsyncOpenAI,
    groq_clients: List[AsyncOpenAI]
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
            gemini_client, groq_clients, api_messages,
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
    groq_clients: List[AsyncOpenAI]
) -> str:
    """
    Generates a concise clarifying question.
    """
    clarify_system = (
        "You are a helpful SHL Assessment Recommender assistant. The user's request is too vague. "
        "Ask a concise, polite clarifying question (strictly 1 sentence) to narrow down their hiring needs. "
        "CRITICAL RULES: "
        "- If the user mentions a contact centre or customer service role, you MUST specifically ask about their preferred call language or regional accent. "
        "- If the user provides a Job Description (JD) or technical stack, you MUST specifically ask whether the role leans backend vs frontend, OR if they need a Senior IC vs Tech Lead. "
        "- Otherwise, ask generally about target role, level of seniority, language requirements, or assessment focus. "
        "Do not output markdown tables or recommendations yet."
    )
    
    api_messages = [{"role": "system", "content": clarify_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        content = await _chat_completion_with_fallback(
            gemini_client, groq_clients, api_messages,
            timeout=settings.CLARIFY_GENERATOR_TIMEOUT,
            temperature=0.3,
            json_mode=False
        )
        return content.strip()
    except Exception as e:
        print(f"Clarifying question generation failed entirely: {e}")
        return "Could you please tell me more about the role you are hiring for and what specific skills or behavioral traits you'd like to assess?"


def _serialize_items_compact(items: List[Dict[str, Any]]) -> str:
    lines = []
    for item in items:
        desc = item.get("description") or ""
        # Truncate description to 120 chars to keep token footprint tiny
        if len(desc) > 120:
            desc = desc[:120].strip() + "..."
        langs = ", ".join(item.get("languages", [])[:3])
        if not langs:
            langs = "—"
        duration = item.get("duration") or "—"
        line = f"[ID:{item['id']}] Name: {item['name']} | Type: {item['test_type']} | Duration: {duration} | Langs: {langs} | Desc: {desc}"
        lines.append(line)
    return "\n".join(lines)


async def generate_response(
    messages: List[Message],
    retrieved_items: List[Dict[str, Any]],
    previous_ids: List[int],
    gemini_client: AsyncOpenAI,
    groq_clients: List[AsyncOpenAI]
) -> Dict[str, Any]:
    """
    Generates the assistant reply and selects recommendations.
    """
    items_text = _serialize_items_compact(retrieved_items)
    response_gen_system = RESPONSE_GENERATOR_SYSTEM_PROMPT + f"\n\nRetrieved Catalog Items:\n{items_text}\n\nPreviously Recommended IDs: {previous_ids}"

    api_messages = [{"role": "system", "content": response_gen_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        content = await _chat_completion_with_fallback(
            gemini_client, groq_clients, api_messages,
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
