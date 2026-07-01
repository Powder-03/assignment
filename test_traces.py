import os
import re
import json
import httpx
import asyncio
from typing import List, Dict, Any

# Target local endpoint
API_URL = "http://127.0.0.1:10000/chat"

def parse_trace_file(file_path: str) -> List[Dict[str, str]]:
    """
    Parses a markdown trace file into a list of user turns.
    Each item contains the user query and the expected response structure.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by Turn sections
    turns_raw = re.split(r"### Turn \d+", content)
    turns = []
    
    for turn_block in turns_raw[1:]:
        # Extract user input
        user_match = re.search(r"\*\*User\*\*\s*\n*\s*>\s*(.*)", turn_block)
        if not user_match:
            continue
        user_query = user_match.group(1).strip()
        
        # Check if this turn expects recommendations
        has_recs = "No recommendations this turn" not in turn_block
        
        # Check if this turn is the end of conversation
        end_of_conv = "`end_of_conversation`: **true**" in turn_block
        
        turns.append({
            "user": user_query,
            "has_recommendations": has_recs,
            "end_of_conversation": end_of_conv
        })
        
    return turns

async def run_trace_test(file_name: str, turns: List[Dict[str, Any]]):
    print(f"\n--- Testing Trace: {file_name} ---")
    messages = []
    
    async with httpx.AsyncClient(timeout=90.0) as client:
        for idx, turn in enumerate(turns, 1):
            messages.append({"role": "user", "content": turn["user"]})
            
            payload = {"messages": messages}
            print(f"Turn {idx} User: {turn['user']}")
            
            # Sleep 15 seconds to avoid Gemini's 15 RPM rate limit across multiple traces
            await asyncio.sleep(15)
            
            try:
                response = await client.post(API_URL, json=payload)
                if response.status_code != 200:
                    print(f"  [FAIL] HTTP {response.status_code}: {response.text}")
                    break
                
                data = response.json()
                reply = data.get("reply", "")
                recs = data.get("recommendations", [])
                end_of_conv = data.get("end_of_conversation", False)
                
                # Print agent reply briefly
                print(f"Agent Reply: {reply[:100]}...")
                print(f"Recommendations: {[r['name'] for r in recs]}")
                print(f"End of Conversation: {end_of_conv}")
                
                # Basic assertions
                if turn["has_recommendations"] and not recs:
                    print(f"  [WARN] Expected recommendations but got none.")
                elif not turn["has_recommendations"] and recs:
                    print(f"  [WARN] Got recommendations but trace did not expect them yet.")
                
                if turn["end_of_conversation"] != end_of_conv:
                    print(f"  [WARN] Expected end_of_conversation={turn['end_of_conversation']}, got {end_of_conv}")
                
                # Append agent reply to messages for conversation history
                messages.append({"role": "assistant", "content": reply})
                
            except Exception as e:
                print(f"  [ERROR] Connection failed: {e}")
                break

async def main():
    traces_dir = "traces"
    if not os.path.exists(traces_dir):
        print(f"Directory {traces_dir} not found.")
        return
        
    trace_files = [os.path.join(traces_dir, f) for f in os.listdir(traces_dir) if f.endswith(".md")]
    trace_files.sort()
    
    for tf in trace_files:
        turns = parse_trace_file(tf)
        await run_trace_test(os.path.basename(tf), turns)

if __name__ == "__main__":
    asyncio.run(main())
