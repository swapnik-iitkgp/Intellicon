from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-large', legacy=False)
model = T5ForConditionalGeneration.from_pretrained('t5-large')

def generate_t5_insights(prompt, relevant_texts):
    combined_prompt = prompt + "\n\n" + "\n\n".join(relevant_texts)
    
    inputs = tokenizer.encode("summarize: " + combined_prompt, return_tensors='pt', max_length=512, truncation=True)
    
    outputs = model.generate(inputs, max_length=300, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response_lines = response.split('. ')
    unique_lines = []
    for line in response_lines:
        if line and line not in unique_lines:
            unique_lines.append(line)
    
    structured_response = '. '.join(unique_lines) + '.'
    
    return structured_response

if __name__ == "__main__":
    prompt = "Compare the risk factors of Google and Tesla."
    relevant_texts = [
        "Risk factors for Google include competition and regulatory scrutiny.",
        "Tesla faces risks such as production delays and regulatory challenges."
    ]
    insight = generate_t5_insights(prompt, relevant_texts)
    print(insight)