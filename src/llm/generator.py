import ollama

def generate_answer(prompt):

    full_answer = ""

    for chunk in ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 150
        },
        stream=True
    ):

        token = chunk["message"]["content"]

        print(token, end="", flush=True)

        full_answer += token

    return full_answer