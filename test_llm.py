# Python talks to the internet
# import "Go and bring this tool into my code"
# #requests A pre-built Python library that knows how to send messages over the internet

import requests

# main function that runs the tests
# from "Go to this specific place"
# deepeval - The DeepEval library installed via pip
# import - Bring this specific thing from there
# evaluate - The specific function that runs all your tests

from deepeval import evaluate

# Specific metric — Answer Relevancy checker
# from "Go to this specific place"
# deepeval - he DeepEval library
# `.` - "Go inside / look inside"
# metrics - A folder inside DeepEval that contains all measurement tools
# import - "Bring this specific thing"
# AnswerRelevancyMetric - The specific tool that measures if an answer is relevant


from deepeval.metrics import AnswerRelevancyMetric


# like a test case template — input + output
# from "Go to this specific place"
# deepeval The DeepEval library
# '.' - "Go inside
# test_case - A folder inside DeepEval for test case templates
# import - "Bring this specific thing"
# LLMTestCase - A template/blueprint to define one test case LLMLarge Language Model (AI model like LLaMA, ChatGPT)TestCaseA single test — input + output bundled together

from deepeval.test_case import LLMTestCase

# lets us connect our own custom model (Groq) to DeepEval
# from - "Go to this specific place"
# deepeval - The DeepEval library
# `.` - "Go inside"
# models - A folder inside DeepEval for AI model connections
# `.` - "Go inside again"
# base_model - A file inside models folder — contains the base template
# import - "Bring this specific thing"
# DeepEvalBaseLLM - A base template/blueprint for connecting ANY custom AI model to DeepEval
# Base - Foundation / starting point
# LLM - Large Language Model

from deepeval.models.base_model import DeepEvalBaseLLM

# Your Groq API Key
# GROQ_API_KEY - A variable name — a box that stores your key
# GROQ - The company whose API we are using
# API - Application Programming Interface — a door to a service
# KEY - The password to open that door
# `=` - Store this value inside this box
# "your_actual_groq_key_here" - The actual password string you got from Groq website

# **Why CAPITAL LETTERS?**
# In Python, writing a variable in ALL CAPS means:
# > *"This is a constant — it never changes throughout the code"*

# It's a convention — like a rule programmers follow to tell each other *"don't change this value!"*

GROQ_API_KEY = "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"

# Step 1 - Create Groq Judge Model (Gemma as Judge)

# We're creating a custom judge class — telling DeepEval
# "Hey, use THIS model to judge answers, not OpenAI!"

# class - "I am creating a new custom blueprint/template"
# GroqJudge - The name we gave our custom blueprint
# Groq - Because it uses Groq's API
# Judge - Because this model's job is to judge answers
# () - "This class is based on / inherits from what's inside these brackets"
# DeepEvalBaseLLM - The blank adapter we imported in line 5
# : - "Here comes the content of this blueprint"

# **What does inherit mean?**
# `DeepEvalBaseLLM` already has some built-in rules about how a judge model should behave. By writing `GroqJudge(DeepEvalBaseLLM)` we are saying:
# > *"Take all those existing rules AND add our Groq-specific stuff on top"*


class GroqJudge(DeepEvalBaseLLM):

    # def - "I am defining a function"
    # init - Special Python function — runs automatically when you CREATE the class
    # __ - Double underscores mean this is a special built-in Python function
    # self - "Refers to this class itself"
    # , - "There is another parameter coming"
    # model_name - A parameter — the name of the Groq model to use as judge

    def __init__(self, model_name):

        # self - "This class itself"
        # . - "Go inside / access something belonging to"
        # model_name - The variable stored INSIDE this class
        # = - "Assign this value"
        # model_name - The value passed in from outside when creating the class

        self.model_name = model_name

# def - "I am defining a function"
# load_model - Function name — DeepEval calls this to know which model to load
# load - "Fetch / bring"
# model - The AI model being used
# self - "This class itself"
#: - "Here comes the content of this function"

    def load_model(self):

        # return - "Send back this value to whoever called this function"
        # self - "This class itself"
        # . - "Access something belonging to"
        # model_name - The model name stored earlier in init

        return self.model_name

    # This is the actual judging — when DeepEval wants to evaluate, it calls Groq's API and gets the judge's opinion back
    # def - "I am defining a function"
    # generate - Function name — this is where actual judging happens
    # self Refers to this class itself
    # prompt: str - Input parameter — the text DeepEval sends to be judged. str means it must be a string (text)
    # -> str "This function will return a string (text)"

    def generate(self, prompt: str) -> str:

        # response - A variable that stores what Groq sends back
        # requests.post - Send a POST request to the internet

        response = requests.post(

            # "https://api.groq.com/..."Groq's API address — same URL you used in Postman!

            "https://api.groq.com/openai/v1/chat/completions",

            # header- Extra information sent with the request

            headers={

                # Authorization - Telling Groq "here is my identity proof
                # f"Bearer {GROQ_API_KEY}" - "Bearer" + your API key — the format Groq expectsf""f-string — lets you put variables inside a string using {}

                "Authorization": f"Bearer {GROQ_API_KEY}",

                # Content-Type - Telling Groq "I am sending JSON format data"
                # "application/json" - The specific format name for JSON

                "Content-Type": "application/json"
            },

            # json={} - The actual data body being sent"
            # model": self.model_nameWhich Groq model to use as judge
            # "messages"The conversation being sent"
            # role": "user" - Telling Groq this message is from a user
            # "content": prompt - The actual text to judge
            # "temperature": 0How creative the response is — 0 means very precise, no randomness

            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0
            }
        )

        # return"Send back this value"
        # response.json() - Convert Groq's response into a Python dictionary
        # ["choices"] - The list of responses Groq returned
        # [0] - Take the first response (index 0)
        # ["message"] - Go inside the message object
        # ["content"] - Get the actual text content

        result = response.json()

        # 👈 This shows us the error

        print(f"\n DEBUG Groq Response: {result}\n")
        return result["choices"][0]["message"]["content"]

    # DeepEval runs tests asynchronously (multiple at once) — this line just says "async version does the same thing as regular version"
# async - "This function can run in the background without blocking other tasks"
# def"I am defining a function"
# a_generateSame as generate but async version
# a_ stands for async - selfRefers to this class itself
# prompt: str - Same prompt input as before
# -> strReturns a string
# return self.generate(prompt) - Just run the regular generate function and return its result
# DeepEval runs multiple test cases at the same time to save time. For that it needs an async version of generate.

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

# def - "I am defining a function"
# get_model_name - Function that returns the model's name
# self - Refers to this class itself
# return - "Send back this value"
# self.model_name - The model name stored during `__init__`

    def get_model_name(self):
        return self.model_name


# Step 2 - Define Answerer and Judge
# Two different models:
# Answerer = big LLaMA model answers your question
# Judge = smaller LLaMA model evaluates the answer
# answerer - A variable storing which model answers questions
# = - "Store this value"
# "llama-3.3-70b-versatile" - Big powerful LLaMA model — the one that answers
# llama - Meta's open source AI model family
# 3.3 - Version 3.3
# 70b - 70 billion parameters — very large and smart
# versatile - Can handle many types of tasks
# judge - A variable storing our custom GroqJudge object
# GroqJudge - The class we built above
# (model_name="llama-3.1-8b-instant") - Tell it to use this specific model
# llama - Meta's open source AI model
# 3.1 - Version 3.1
# 8b - 8 billion parameters — smaller and faster than 70b
# instant - Optimized for speed

answerer = "llama-3.3-70b-versatile"

# judge - A variable that stores our custom judge object
# = - "Store this value"
# GroqJudge - The custom class we built above
# () - "Create a new object from this class"
# model_name - The parameter defined in init
# = - "The value going into model_name is"
# "llama-3.1-8b-instant" - Actual value going in 👇

judge = GroqJudge(model_name="llama-3.1-8b-instant")

# Step 3 - Get Answer from LLaMA
# This sends your question to LLaMA and gets back the answer
# def - "I am defining a function"
# get_llm_answer - Name of the function
# get - "Fetch / retrieve"
# llm - Large Language Model
# answer - The response from the model
# prompt - Parameter — the question going in


def get_llm_answer(prompt):

    # response - Variable that stores whatever Groq sends back
    # = - "Store this value"
    # requests - The internet library we imported
    # . - "Access something inside requests"
    # post - Send a POST request — exactly like Postman!
    # ( - "Parameters coming below"

    response = requests.post(

        # This is the URL parameter going into requests.post

        "https://api.groq.com/openai/v1/chat/completions",

        # headers - Extra info sent with request — like an envelope label
        # { } - A dictionary — key value pairs
        # "Authorization" - Proving your identity to Groq
        # : - "Value for this key is"
        # f"" - f-string — lets you insert variables using {}
        # Bearer - Word Groq expects before the API key
        # {GROQ_API_KEY} - Actual value going in
        # "Content-Type" - Telling Groq what format data is coming in
        # "application/json" - Actual value going in

        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },

        # json - The body of the request — same as Body tab in Postman!
        # { } - A dictionary
        # "model" - Which model should answer the question
        # : - "Value for this key is"
        # answerer - Variable we defined earlier

        json={
            "model": answerer,

            # "messages" - The conversation being sent to LLaMA
            # [ ] - A list — can hold multiple messages
            # { } - Each message is a dictionary
            # "role" - Who is sending this message
            # : - "Value for this key is"
            # "user": Actual value going in
            # "content" - The actual text of the message
            # : - "Value for this key is"
            # prompt - The question passed into get_llm_answer()
            # "temperature" - Controls how creative/random the response is
            # : - "Value is"
            # What this means:
            # temperature = 0
            #   0   = No randomness — always precise, consistent answer
            #   0.5 = Some creativity
            #   1.0 = Very creative and random
            #   For testing we always use 0 — we need consistent results!

            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        }
    )

    # return - "Send back this value to whoever called this function"
    # response - The reply we got back from Groq
    # .json() - Convert Groq's raw response into readable Python dictionary
    # ["choices"] - Go to the "choices" key
    # [0] - Take the FIRST response from the list
    # ["message"] - Go inside the message object
    # ["content"] - Get the actual text

    return response.json()["choices"][0]["message"]["content"]


# Step 4 - Define Your Test
# input_prompt = the question you asked
# actual_output = the answer LLaMA gave back

input_prompt = "What is software testing?"
actual_output = get_llm_answer(input_prompt)

# print - Show this text on the screen
# f"" - f-string — lets you insert variables using {}
# \n - New line — moves to next line
# " LLaMA Answer:" - Text that prints as label
# \n - Another new line
# {actual_output} - Actual value

print(f"\n LLaMA Answer:\n{actual_output}\n")

# Step 5 - Evaluate with Gemma as Judge
# input = Test Step ("Ask: What is software testing?")
# actual_output = Actual Result ("LLaMA's answer")

test_case = LLMTestCase(
    input=input_prompt,
    actual_output=actual_output
)

# Setting up the Metric
# threshold=0.7 = minimum 70% relevancy score to PASS ✅
# model=judge = use Gemma as the judge
# include_reason=True = tell me why it passed or failed
# metric - Variable storing the evaluation rules
# = - "Store this value"
# AnswerRelevancyMetric - The relevancy checker we imported
# () - "Create a new metric object"
# threshold - Minimum score to pass
# = - "Value going in is"
# model - Which model to use as judge
#   True  = "Tell me WHY it passed or failed"
#   False = "Just give me the score"
#   We use True so we can learn from the results!

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True
)

# Running the Evaluation
# evaluate - The main function that runs everything
# () - "Parameters coming"
# test_cases - List of all test cases to run
# = - "Value going in is"
# Your complete test case goes inside this list
# [ ] means you can add multiple test cases later!

evaluate(
    test_cases=[test_case],
    metrics=[metric]
)
