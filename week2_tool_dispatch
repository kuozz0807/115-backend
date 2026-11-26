import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / "APIKEY.env"
load_dotenv(dotenv_path=env_path)

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m3.1-flash"


def call_model(messages, tools=None):
    api_key = os.environ["NVIDIA_API_KEY"]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 1,
        "top_p": 0.95,
    }
    if tools is not None:
        payload["tools"] = tools

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def dispatch(name, arguments):
    if name != "http_get":
        raise ValueError(f"unknown tool: {name}")
    args = json.loads(arguments)
    resp = requests.get(args["url"], timeout=10)
    return {
        "status_code": resp.status_code,
        "body_summary": resp.text[:120],
    }


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "http_get",
                "description": "Fetch a URL with HTTP GET. Return status code and a short body summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute http(s) URL to fetch.",
                        }
                    },
                    "required": ["url"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "用 http_get 抓取 https://example.com，然後用一句話說明你看到什麼。",
        }
    ]

    first_data = call_model(messages, tools=tools)
    first = first_data["choices"][0]["message"]

    if not first.get("tool_calls"):
        print("模型未呼叫工具，直接回覆:", first.get("content"))
    else:
        call = first["tool_calls"][0]
        print("模型建議:", call["function"]["arguments"])
        result = dispatch(call["function"]["name"], call["function"]["arguments"])
        print("程式執行:", json.dumps(result, ensure_ascii=False))

        messages.append(first)
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(result, ensure_ascii=False),
        })

        print("\n回填後的 messages:")
        print(json.dumps(messages, ensure_ascii=False, indent=2))

        second_data = call_model(messages, tools=None)
        final_content = second_data["choices"][0]["message"]["content"]
        print("\n最終回覆:", final_content)
