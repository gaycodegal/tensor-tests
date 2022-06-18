from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
TFGPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)
input_ids = tokenizer.encode("My boyfriend has taken up residence in another family member. Which one?", return_tensors='tf')
generated_text_samples = model.generate(
    input_ids, 
    max_length=150,  
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,
    top_p=0.92,
    temperature=.85,
    do_sample=True,
    top_k=125,
    early_stopping=True
)
#Print output for each sequence generated above
for i, beam in enumerate(generated_text_samples):
  print("{}: {}".format(i,tokenizer.decode(beam, skip_special_tokens=True)))
  print()


