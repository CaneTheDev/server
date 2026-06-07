import httpx
import json
import base64

def test_cerebras_vision():
    api_key = "csk-c6k85me2nr4r6kdvpwf5k8wwtmcppc6yhwpwejjv6j9jemj2"
    url = "https://api.cerebras.ai/v1/chat/completions"
    # Using a different image URL that is less likely to block simple requests
    image_url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Step 1: Download the image and encode to Base64
    print(f"Downloading image from {image_url}...")
    try:
        # Added a User-Agent to avoid 403 Forbidden errors
        client_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        with httpx.Client(headers=client_headers) as client:
            img_response = client.get(image_url)
            img_response.raise_for_status()
            base64_image = base64.b64encode(img_response.content).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_image}"
            print("Successfully encoded image to Base64.")
    except Exception as e:
        print(f"Failed to download or encode image: {str(e)}")
        return

    payload = {
        "model": "gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ]
    }
    
    print(f"Testing Cerebras vision with model: {payload['model']} using Base64 data URI...")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            print(f"Status Code: {response.status_code}")
            print("Response Body:")
            if response.status_code == 200:
                print(json.dumps(response.json(), indent=2))
            else:
                print(response.text)
    except Exception as e:
        print(f"Error occurred during API call: {str(e)}")

if __name__ == "__main__":
    test_cerebras_vision()
