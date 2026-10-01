# Example 4.24
from transformers import pipeline

question_answerer = pipeline('question-answering', framework='pt')
print(question_answerer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
}))
