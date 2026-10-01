# Exercise 4.19: Compare sentiment for positive and negative sentences.
# pip install transformers torch
from transformers import pipeline

classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    framework='pt',
)

sentences = [
    'The service was quick and the staff were kind.',
    'The instructions were confusing and the device stopped working.',
]

for sentence in sentences:
    result = classifier(sentence)[0]
    print(sentence)
    print(f"Predicted label: {result['label']}")
    print(f"Confidence: {result['score']:.4f}\n")

# Compare each predicted label and confidence score with the sentence sentiment.
