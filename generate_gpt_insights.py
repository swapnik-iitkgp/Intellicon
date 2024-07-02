from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_gpt_insights(prompt, relevant_texts):
    combined_prompt = prompt + "\n\n" + "\n\n".join(relevant_texts)
    
    inputs = tokenizer.encode(combined_prompt, return_tensors='pt')
    
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.replace(prompt, '').strip()
    
    response_lines = response.split('. ')
    unique_lines = []
    for line in response_lines:
        if line and line not in unique_lines and not line.endswith(','):
            unique_lines.append(line)
    
    structured_response = '. '.join(unique_lines) + '.'
    
    return structured_response

if __name__ == "__main__":
    prompt = "Compare the risk factors of Google and Tesla."
    relevant_texts = [
        "Risk factors for Google include competition and regulatory scrutiny.",
        "Tesla faces risks such as production delays and regulatory challenges."
    ]
    insight = generate_gpt_insights(prompt, relevant_texts)
    print(insight)