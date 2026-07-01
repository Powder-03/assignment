import json
import faiss
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from shl_recommender.domain.schemas import ChatRequest, ChatResponse, RecommendationItem, Message
from shl_recommender.services import catalog as catalog_svc
from shl_recommender.services import search as search_svc
from shl_recommender.services import llm as llm_svc

async def run_recommender_pipeline(
    request: ChatRequest,
    catalog: Dict[int, Dict[str, Any]],
    faiss_index: faiss.IndexIDMap,
    gemini_client: AsyncOpenAI,
    groq_clients: List[AsyncOpenAI],
    embed_client: AsyncOpenAI
) -> ChatResponse:
    """
    Executes the stateless chat recommender pipeline:
    1. Turn-cap guardrail checks.
    2. Previous recommendations state recovery.
    3. State extraction (vagueness, out of scope, constraints extraction) on Groq (aclient).
    4. Pre-filtered catalog retrieval (using FAISS + OpenRouter embed_client).
    5. Response generation grounding on Groq (aclient).
    6. Markdown table formatting and schema validation.
    """
    # Safe fallback
    fallback_response = ChatResponse(
        reply="I apologize, I encountered a temporary issue. Could you please describe what role or skills you are hiring for?",
        recommendations=[],
        end_of_conversation=False
    )
    
    try:
        # 0. Guard against empty or whitespace-only user messages
        if not request.messages or not request.messages[-1].content.strip():
            return fallback_response

        # 1. Turn-cap check (spec: max 8 total messages including user & assistant)
        total_turns = len(request.messages)
        turn_limit_reached = total_turns >= 7  # At 7 messages, we have room for exactly 1 reply to hit 8

        # Recover previous shortlist IDs from messages history
        previous_ids = catalog_svc.parse_previous_recommendations(request.messages, catalog)

        # 2. State Extractor (Step 1)
        state = await llm_svc.extract_state(request.messages, turn_limit_reached, gemini_client, groq_clients)
        print(f"[DEBUG] Extracted state: {json.dumps(state, indent=2)}")

        if turn_limit_reached:
            state["recommendations_ready"] = True

        # Process out-of-scope refusals
        if state.get("is_out_of_scope"):
            out_of_scope_recs = []
            for sid in previous_ids:
                if sid in catalog:
                    item = catalog[sid]
                    out_of_scope_recs.append(RecommendationItem(
                        id=item["id"],
                        name=item["name"],
                        test_type=item["test_type"],
                        keys=item.get("keys", []),
                        duration=item.get("duration"),
                        languages=item.get("languages", []),
                        url=item.get("url", ""),
                        description=item.get("description")
                    ))
            
            out_of_scope_reply = "I can help you select and recommend SHL assessments from the catalog, but I cannot assist with general HR advice, legal compliance questions, or off-topic prompts. Please let me know what kinds of skills or roles you are hiring for, and I will find the right assessments."
            if out_of_scope_recs:
                hidden_state = "<!-- State: " + ", ".join([str(r.id) for r in out_of_scope_recs]) + " -->"
                out_of_scope_reply += f"\n\n{hidden_state}"
            
            return ChatResponse(
                reply=out_of_scope_reply,
                recommendations=[],
                end_of_conversation=False
            )

        # If recommendations are not ready, we must clarify.
        if not state.get("recommendations_ready") and not turn_limit_reached:
            reply_content = await llm_svc.generate_clarifying_question(request.messages, gemini_client, groq_clients)
            return ChatResponse(
                reply=reply_content,
                recommendations=[],
                end_of_conversation=False
            )

        # 3. Pre-Filtered Retrieval (Step 2)
        allowed_test_types = state.get("allowed_test_types", [])
        
        # Filter items by test type intersection
        allowed_ids = []
        allowed_set = set(allowed_test_types)
        for item_id, item in catalog.items():
            item_letters = [x.strip() for x in item["test_type"].split(",") if x.strip()]
            if not allowed_set or (set(item_letters) & allowed_set):
                allowed_ids.append(item_id)
        
        if not allowed_ids:
            allowed_ids = list(catalog.keys())

        # Perform semantic index search using OpenRouter embed_client
        semantic_search_term = state.get("semantic_search_term", "").strip()
        
        query_embedding = None
        if semantic_search_term:
            query_embedding = await search_svc.get_query_embedding(semantic_search_term, embed_client)

        # Retrieve nearest items
        retrieved_ids = search_svc.search_index(query_embedding, allowed_ids, faiss_index)

        # Resolve items to compare
        compare_ids = []
        for name in state.get("items_to_compare", []):
            name_lower = name.lower().strip()
            for item_id, item in catalog.items():
                if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                    compare_ids.append(item_id)
        
        # Append comparison items
        for cid in compare_ids:
            if cid not in retrieved_ids:
                retrieved_ids.append(cid)

        # Merge with previously recommended IDs to ensure the generator can review them
        combined_ids = list(retrieved_ids)
        for prev_id in previous_ids:
            if prev_id not in combined_ids:
                combined_ids.append(prev_id)

        retrieved_items = [catalog[rid] for rid in combined_ids if rid in catalog]
        print(f"[DEBUG] Retrieved items: {[item['name'] for item in retrieved_items]}")
        print(f"[DEBUG] Previous recommended IDs: {previous_ids}")

        # 4. Response Generator (Step 3)
        gen_response = await llm_svc.generate_response(request.messages, retrieved_items, previous_ids, gemini_client, groq_clients)
        print(f"[DEBUG] Generator response: {json.dumps(gen_response, indent=2)}")

        # 5. Reconstruct and Format Response (Step 4)
        selected_ids_raw = gen_response.get("selected_ids", [])
        
        # Validate selected IDs exist in catalog and deduplicate
        selected_ids = []
        seen = set()
        for sid in selected_ids_raw:
            if sid in catalog and sid not in seen:
                seen.add(sid)
                selected_ids.append(sid)

        # Hard cap at 10 recommendations (spec requirement)
        selected_ids = selected_ids[:10]

        recommendations_ready = state.get("recommendations_ready", False)
        
        if not recommendations_ready or not selected_ids:
            reply = gen_response.get("assistant_reply", "").strip()
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=state.get("end_of_conversation", False)
            )

        # Format recommendations list of objects
        recommendations_list = []
        for sid in selected_ids:
            item = catalog[sid]
            recommendations_list.append(RecommendationItem(
                id=item["id"],
                name=item["name"],
                test_type=item["test_type"],
                keys=item.get("keys", []),
                duration=item.get("duration"),
                languages=item.get("languages", []),
                url=item.get("url", ""),
                description=item.get("description")
            ))

        final_reply = gen_response.get("assistant_reply", "").strip()

        # Append hidden IDs to the reply string so the stateless state recovery
        # (parse_previous_recommendations) can find them in the next turn's history.
        if recommendations_list:
            hidden_state = "<!-- State: " + ", ".join([str(r.id) for r in recommendations_list]) + " -->"
            final_reply += f"\n\n{hidden_state}"

        # Only allow end_of_conversation if we actually have a shortlist
        eoc = state.get("end_of_conversation", False) and len(recommendations_list) > 0

        return ChatResponse(
            reply=final_reply,
            recommendations=recommendations_list,
            end_of_conversation=eoc
        )

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return fallback_response
