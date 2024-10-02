import boto3
import json

prompt_data = """
tell us more about SAP modules
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt": f"<s>[INST] {prompt_data} [/INST]",
    "max_tokens": 3072,  
    "temperature": 0.5,  
    "top_p": 0.9,        
    "top_k": 50          
}

body = json.dumps(payload)

model_id = "mistral.mistral-large-2402-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    response_body_str = response['body'].read().decode('utf-8')
    print("Full response body:", response_body_str)

    response_body = json.loads(response_body_str)
    print("Parsed response body:", response_body)
    if 'outputs' in response_body:
        response_text = response_body['outputs'][0]['text']
        print("Generated text:", response_text)
    else:
        print("Key 'outputs' not found in the response. Response structure might be different.")

except Exception as e:
    print("An error occurred:", str(e))
