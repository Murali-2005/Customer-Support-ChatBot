import os
import base64
import mimetypes
from PIL import Image
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


def analyze_image(image_path: str):

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = """
You are an intelligent multimodal reasoning assistant.

Analyze the uploaded image carefully.

Return ONLY in the following format.

Summary:
<one paragraph>

Objects:
- ...

Visible Text (OCR):
- ...

Important Observations:
- ...

Possible Issues:
- ...

Evidence:
- Mention the exact parts of the image supporting your conclusions.

Ambiguity:
- If anything is unclear, explain why.

Confidence:
- High / Medium / Low

Do not guess.

If the image quality is poor or information is insufficient,
explicitly state what additional information is needed.
"""

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]
    )

    response = llm.invoke([message])

    return {
        "analysis": response.content,
        "image_used": True
    }


def get_image_context(image_path: str):

    result = analyze_image(image_path)

    return result["analysis"]