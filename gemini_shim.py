
import os
import json
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

# Configuration
TARGET_URL = "http://localhost:8001/v1/chat/completions"
MODELS_URL = "http://localhost:8001/v1/models"
AUTH_TOKEN = "1234"
DEFAULT_MODEL = "MLX-Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-bf16"

def map_to_gemini_model(local_id):
    # Gemini CLI expects model names to start with 'models/'
    return {
        "name": f"models/{local_id}",
        "version": "local",
        "displayName": f"Local: {local_id}",
        "description": "Local model via Spoke shim",
        "supportedGenerationMethods": ["generateContent", "countTokens"]
    }

@app.get("/v1beta/models")
@app.get("/v1/models")
async def list_models():
    print("Intercepted: List Models")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                MODELS_URL,
                headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                timeout=10.0
            )
            local_models = resp.json().get("data", [])
            gemini_models = [map_to_gemini_model(m["id"]) for m in local_models]
            # Also include some fake standard ones to keep the CLI happy
            return {"models": gemini_models}
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"models": [map_to_gemini_model(DEFAULT_MODEL)]}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(request: Request, path: str):
    print(f"Intercepted: {request.method} {path}")
    
    if "generateContent" in path:
        body = await request.json()
        
        # Extract model from path or body
        requested_model = DEFAULT_MODEL
        for m in [path.split("/")[-1].split(":")[0]]:
            if m and m != "models":
                requested_model = m.replace("models/", "")

        messages = []
        for content in body.get("contents", []):
            role = "assistant" if content.get("role") == "model" else "user"
            text = "".join([part.get("text", "") for part in content.get("parts", [])])
            messages.append({"role": role, "content": text})

        is_streaming = "streamGenerateContent" in path
        
        openai_payload = {
            "model": requested_model,
            "messages": messages,
            "stream": is_streaming
        }

        if not is_streaming:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    TARGET_URL,
                    json=openai_payload,
                    headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                    timeout=60.0
                )
            data = resp.json()
            return {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": data["choices"][0]["message"]["content"]}]
                    },
                    "finishReason": "STOP"
                }]
            }
        else:
            async def stream_translator():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST", 
                        TARGET_URL, 
                        json=openai_payload, 
                        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
                        timeout=60.0
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            if "[DONE]" in line:
                                break
                            try:
                                chunk = json.loads(line[6:])
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield b"[" + json.dumps({
                                        "candidates": [{
                                            "content": {
                                                "role": "model",
                                                "parts": [{"text": content}]
                                            }
                                        }]
                                    }).encode() + b"]\r\n"
                            except:
                                continue
            return StreamingResponse(stream_translator(), media_type="application/json")

    return {"status": "ok", "message": "Proxied by Gemini Shim"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8888)
