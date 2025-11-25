from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
client = OpenAI(api_key=SECRET_KEY)

tags = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']

sentence = input("Enter sentence : ")
prompt = f"Tag each word in the sentence with its POS tag {tags} in the format: (word : tag).\nSentence: {sentence}"

resp = client.chat.completions.create(
    model="gpt-4.1-mini", 
    messages=[{"role": "user", "content": prompt}]
)

print(resp.choices[0].message.content)
