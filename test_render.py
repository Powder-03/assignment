import httpx
import json
import asyncio

async def main():
    url = "https://assignment-j95o.onrender.com/chat"
    payload = {
      "messages": [
        {
          "role": "user",
          "content": "We need a solution for senior leadership."
        },
        {
          "role": "assistant",
          "content": "Could you please specify what type of senior leadership role you are looking to assess?"
        },
        {
          "role": "user",
          "content": "The pool consists of CXOs, director-level postions; people with more than 15 years of experience."
        },
        {
          "role": "assistant",
          "content": "For these highly experienced senior leaders, are you looking to assess their strategic decision-making... is this for a newly created position, or developmental feedback for an executive already in role?"
        },
        {
          "role": "user",
          "content": "Selection — comparing candidates against a leadership benchmark."
        }
      ]
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"Sending POST request to {url}...")
        response = await client.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    asyncio.run(main())
