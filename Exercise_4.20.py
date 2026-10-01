# Exercise 4.20: Ask different questions about a different context.
# pip install transformers torch
from transformers import pipeline

question_answerer = pipeline(
    'question-answering',
    model='distilbert-base-cased-distilled-squad',
    framework='pt',
)

context = (
    'Orchid Labs was founded by Maya Patel in Singapore in 2019. '
    'The company develops software for hospitals.'
)
questions = [
    'Who founded Orchid Labs?',
    'Where was Orchid Labs founded?',
    'When was Orchid Labs founded?',
]

for question in questions:
    result = question_answerer(question=question, context=context)
    print(question)
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.4f}\n")
