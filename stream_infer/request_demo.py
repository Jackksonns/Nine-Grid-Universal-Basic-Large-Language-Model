import json
import pprint
import requests
import sseclient # pip install sseclient-py



# content = "hello"


url = "http://10.1.2.1:8888/llm"

payload = json.dumps({
  "content": "<用户>hello<AI>"
})
headers = {
  'Content-Type': 'application/json',
  "accept": "text/event-stream"
}

response = requests.request("POST", url, stream=True, headers=headers, data=payload)

# print(response.text)


client = sseclient.SSEClient(response)

for event in client.events():
    pprint.pprint(event.data)