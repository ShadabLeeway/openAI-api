# import openai
# import os


# # Set OpenAI API key
# os.environ["OPENAI_API_KEY"]="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"

# # Set model name and parameters
# model_name = "text-davinci-002"
# temperature = 0.7
# max_tokens = 60
# openai.api_key = os.environ["OPENAI_API_KEY"]

# # Set prompt
# # prompt = input("Enter the query \n")
# # Call OpenAI API
# def func(prompt):
#     response = openai.Completion.create(
#     engine=model_name,
#     prompt=prompt,
#     temperature=temperature,
#     max_tokens=max_tokens)
    
#     return response

# # Process response
# response=func("What is apple")
# answer = response.choices[0].text.strip() # type: ignore 
# # Print answer
# print(answer)



import openai
API_KEY = "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"
openai.api_key = API_KEY
def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
    )
    return response.choices[0].text.strip()
prompt = input("Enter some input \n")
response = generate_text(prompt)
print(response)