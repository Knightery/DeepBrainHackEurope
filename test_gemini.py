from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Missing GEMINI_API_KEY in .env")
        return 1

    prompt = "Reply with exactly: Gemini test OK"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = (getattr(response, "text", "") or "").strip()

    print("Model: gemini-2.5-flash")
    print("Prompt:", prompt)
    print("Response:", text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

