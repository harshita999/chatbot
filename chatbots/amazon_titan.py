import boto3
import json

prompt_data = """
tell us more about SAP modules
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "inputText": prompt_data,
    "textGenerationConfig": {
        "maxTokenCount": 3072,
        "stopSequences": [],  
        "temperature": 0.7,
        "topP": 0.9
    }
}

body = json.dumps(payload)

model_id = "amazon.titan-text-premier-v1:0"

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
    if 'results' in response_body:
        response_text = response_body['results'][0]['outputText']
        print("Generated text:", response_text)
    else:
        print("Key 'results' not found in the response. Response structure might be different.")

except Exception as e:
    print("An error occurred:", str(e))
