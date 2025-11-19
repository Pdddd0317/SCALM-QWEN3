from deeplake.core.vectorstore import VectorStore
import glob
import os
import requests
import json


os.environ['OPENAI_API_KEY'] = "sk-5T7lKz5RaRLetIB9"

# Vector Store data structure can be summarized using vector_store.summary()

vector_store_path = r'database\DApp_extr'
vector_store = VectorStore(
        path=vector_store_path,
    )


def create_database(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        code = file.read()

    CHUNK_SIZE = 1000
    chunked_code = [code[i:i + 1000] for i in range(0, len(code), CHUNK_SIZE)]

    vector_store.add(text=chunked_code,
                     embedding_function=embedding_function,
                     embedding_data=chunked_code,
                     metadata=[{"path": file_path}] * len(chunked_code))


def query_database(query):
    search_results = vector_store.search(embedding_data=query, embedding_function=embedding_function)
    return search_results['text'][0]


def embedding_function(texts, model="text-embedding-ada-002"):
    if isinstance(texts, str):
        texts = [texts]
    # Replace newlines with spaces
    texts = [t.replace("\n", " ") for t in texts]

    # Set request headers
    headers = {
        'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
        'Content-Type': 'application/json',
    }
    # Build request body
    payload = {
        'input': texts,
        'model': model,
    }
    # Send request to OpenAI API
    response = requests.post('https://api.zongwei.cc/v1/embeddings',
                             headers=headers,
                             data=json.dumps(payload))
    # Check response status code
    if response.status_code == 200:
        # Parse response data and return embedding vectors
        response_json = response.json()
        return [data['embedding'] for data in response_json['data']]
    else:
        # If error occurs, print error message
        print("Error:", response.status_code, response.text)
        return None


if __name__ == '__main__':
    print(vector_store_path)
    # answers = query_database("SWC-Unchecked_Call_Return_Value")
    # print(answers)
    # for answer in answers:
    #     print("########################")
    #     print(answer)

    # Get all file paths ending with 'graph.dot'
    # dot_file_paths = glob.glob(os.path.join(r"C:\Users\33519\Desktop\DAppSCAN\extracted_SWCs", "**", "*_extracted_extracted.sol"), recursive=True)
    #
    # # Iterate through file paths and print them
    # for file_path in dot_file_paths:
    #     print(file_path)
    #     create_database(file_path)