import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from shl_recommender.domain.schemas import Message
from shl_recommender.core.config import settings
from shl_recommender.core.constants import STATE_EXTRACTOR_SYSTEM_PROMPT, RESPONSE_GENERATOR_SYSTEM_PROMPT

async def extract_state(
    messages: List[Message],
    turn_limit_reached: bool,
    aclient: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Calls Groq Cloud API using aclient to extract structured conversation state.
    """
    system_prompt = STATE_EXTRACTOR_SYSTEM_PROMPT
    if turn_limit_reached:
        system_prompt += "\n\nIMPORTANT: Turn limit reached. You MUST set 'recommendations_ready' to true and produce a best-effort recommendation from current context."

    api_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        response = await asyncio.wait_for(
            aclient.chat.completions.create(
                model=settings.GROQ_LLM_MODEL,
                messages=api_messages,
                response_format={"type": "json_object"},
                temperature=0.0
            ),
            timeout=settings.STATE_EXTRACTOR_TIMEOUT
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"State extractor failed: {e}")
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

async def generate_clarifying_question(messages: List[Message], aclient: AsyncOpenAI) -> str:
    """
    Calls Groq Cloud API using aclient to generate a concise clarifying question (max 2 sentences).
    """
    clarify_system = "You are a helpful SHL Assessment Recommender assistant. The user's request is too vague. Ask a concise, polite clarifying question (at most 2 sentences) to narrow down their hiring needs, such as target role, level of seniority, language requirements, or assessment focus. Do not output markdown tables or recommendations yet."
    
    api_messages = [{"role": "system", "content": clarify_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        response = await asyncio.wait_for(
            aclient.chat.completions.create(
                model=settings.GROQ_LLM_MODEL,
                messages=api_messages,
                temperature=0.7
            ),
            timeout=settings.CLARIFY_GENERATOR_TIMEOUT
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Clarifying question generation failed: {e}")
        return "Could you please tell me more about the role you are hiring for and what specific skills or behavioral traits you'd like to assess?"

async def generate_response(
    messages: List[Message],
    retrieved_items: List[Dict[str, Any]],
    previous_ids: List[int],
    aclient: AsyncOpenAI
) -> Dict[str, Any]:
    """
    Calls Groq Cloud API using aclient to generate the assistant reply and select recommendations.
    """
    items_json_str = json.dumps(retrieved_items, indent=2, ensure_ascii=False)
    response_gen_system = RESPONSE_GENERATOR_SYSTEM_PROMPT + f"\n\nRetrieved Catalog Items:\n{items_json_str}\n\nPreviously Recommended IDs: {previous_ids}"

    api_messages = [{"role": "system", "content": response_gen_system}]
    for m in messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        response = await asyncio.wait_for(
            aclient.chat.completions.create(
                model=settings.GROQ_LLM_MODEL,
                messages=api_messages,
                response_format={"type": "json_object"},
                temperature=0.0
            ),
            timeout=settings.RESPONSE_GENERATOR_TIMEOUT
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Response generator failed: {e}")
        return {
            "assistant_reply": "I have updated the recommended shortlist based on your requirements.",
            "selected_ids": previous_ids if previous_ids else [item["id"] for item in retrieved_items[:3]]
        }
