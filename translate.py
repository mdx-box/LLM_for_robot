from openai import OpenAI
client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    # sk-xxx替换为自己的key
    api_key='sk-F6WswezCoMSMY6aofvkILDKUcxOjM44qmZLOm7Sfhl0gIc7d'
)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])