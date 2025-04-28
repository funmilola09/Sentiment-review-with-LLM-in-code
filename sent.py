#import sys
#sys.path.append('/c:/Users/Admin/Desktop/projects/Sentiment review with LLM in code/')  
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_response(prompt, model="gpt-3.5-turbo"):
    """
    Function to get a response from the OpenAI API using the specified model.
    
    Parameters:
    - prompt (str): The input prompt for the model.
    - model (str): The model to use for generating the response. Default is "gpt-3.5-turbo".
    
    Returns:
    - str: The generated response from the model.
    """
    from openai import OpenAI; client = openai.OpenAI(api_key="your_api_key_here")
    client.chat.completions.create(
    #response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return client.choices[0].message['content']

#define a prompt that will classify the sentiment of a resturant review
prompt = """ classify the sentiment of the following review as positive, negative or neutral.
The banana pudding was delicious.
"""
#call the llm_response function to get the sentiment of the review
response = llm_response(prompt)
print(response) # print the response from the model