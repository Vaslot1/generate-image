import base64
import requests
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash-image-preview"
PROMPT_FILE = 'prompt.txt'
RESULT_DIR = 'result'
SOURCE_DIR = 'source'


def get_prompt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_from_image(prompt: str, image_base64: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        url=f"{OPENROUTER_API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            "modalities": ["image", "text"],
        }
    )

    response.raise_for_status()
    data = response.json()

    generated_image_data = data['choices'][0]['message']['images'][0]['image_url']['url']

    if "," in generated_image_data:
        generated_image_data = generated_image_data.split(',')[1]
        
    return generated_image_data

def save_decoded_image(image_data_base64: str, output_path: str):
    image_data = base64.b64decode(image_data_base64)
    with open(output_path, 'wb') as file:
        file.write(image_data)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <image_name_in_source_folder>")
        return

    source_image_name = sys.argv[1]
    source_image_path = os.path.join(SOURCE_DIR, source_image_name)

    if not os.path.exists(source_image_path):
        print(f"Error: Image '{source_image_name}' not found in '{SOURCE_DIR}' directory.")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)
    
    try:
        prompt = get_prompt(PROMPT_FILE)
        
        print(f"Encoding source image: {source_image_path}")
        image_base64 = encode_image_to_base64(source_image_path)
        
        print(f"Generating new image with model {MODEL}...")
        generated_image_data = generate_image_from_image(prompt, image_base64)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(source_image_name)

        if not ext:
            ext = '.png'
        result_image_name = f"{base_name}_{timestamp}{ext}"
        result_image_path = os.path.join(RESULT_DIR, result_image_name)
        
        print(f"Saving generated image to: {result_image_path}")
        save_decoded_image(generated_image_data, result_image_path)
        
        print(f"Deleting source image: {source_image_path}")
        os.remove(source_image_path)
        
        print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
