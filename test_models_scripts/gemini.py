from transformers import pipeline, set_seed

summarizer = pipeline("text2text-generation", model="describeai/gemini")
code = "print('hello world!')"

response = summarizer(code, max_length=100, num_beams=3)
# print("Summarized code: " + response[0]['generated_text'])
print(response[0]["generated_text"])

# works

# AIzaSyA9o98h93t0KH-bmInn-nZI3MEvjTzWB5k
