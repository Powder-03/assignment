import re
import json
from typing import List, Dict, Any
from shl_recommender.domain.schemas import Message

def load_catalog(catalog_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Loads catalog from clean_catalog.json and returns a dict mapped by integer ID.
    """
    try:
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog_list = json.load(f)
        return {item["id"]: item for item in catalog_list}
    except Exception as e:
        print(f"Error loading catalog file: {e}")
        return {}

def parse_previous_recommendations(messages: List[Message], catalog: Dict[int, Dict[str, Any]]) -> List[int]:
    """
    Parses the conversation history backwards to recover the last recommended shortlist.
    Uses hidden state comments (e.g. <!-- State: 105, 212 -->) to resolve IDs.
    """
    state_pattern = re.compile(r"<!--\s*State:\s*([\d,\s]+)\s*-->")
    
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
            
        match = state_pattern.search(msg.content)
        if match:
            id_str = match.group(1)
            found_ids = []
            for token in id_str.split(","):
                token = token.strip()
                if token.isdigit():
                    fid = int(token)
                    if fid in catalog:
                        found_ids.append(fid)
            
            if found_ids:
                # Remove duplicates while preserving order
                seen = set()
                ordered_ids = []
                for fid in found_ids:
                    if fid not in seen:
                        seen.add(fid)
                        ordered_ids.append(fid)
                return ordered_ids
                
    return []

def format_languages(languages_list: List[str]) -> str:
    """
    Formats languages list for markdown table representation:
    If > 4 languages, show first 4 and append " _(+N more)_".
    """
    if not languages_list:
        return "—"
    if len(languages_list) <= 4:
        return ", ".join(languages_list)
    else:
        first_4 = ", ".join(languages_list[:4])
        remaining = len(languages_list) - 4
        return f"{first_4} _(+{remaining} more)_"

def build_markdown_table(ids: List[int], catalog: Dict[int, Dict[str, Any]]) -> str:
    """
    Builds the markdown table of recommended assessments.
    """
    header = "| # | Name | Test Type | Keys | Duration | Languages | URL |\n"
    separator = "|---|------|-----------|------|----------|-----------|-----|\n"
    rows = []
    for idx, sid in enumerate(ids, 1):
        item = catalog[sid]
        name = item["name"]
        test_type = item["test_type"]
        keys_str = ", ".join(item.get("keys", []))
        duration = item.get("duration") or "—"
        if duration == "-":
            duration = "—"
        languages = format_languages(item.get("languages", []))
        url = f"<{item['url']}>"
        rows.append(f"| {idx} | {name} | {test_type} | {keys_str} | {duration} | {languages} | {url} |")
    return header + separator + "\n".join(rows)
