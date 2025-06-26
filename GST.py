
!pip install transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Load GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
# Set topic prompt
topic = "MS Dhoni is one of the most successful captains in cricket history. He is known for his calm demeanor and"
# Encode input
input_ids = tokenizer.encode(topic, return_tensors='pt')
# Generate text
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        early_stopping=True
    )
# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text about MS Dhoni (GPT-2):\n", generated_text)
