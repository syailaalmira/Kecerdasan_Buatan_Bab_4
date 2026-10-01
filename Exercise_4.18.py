# Exercise 4.18: Analyze a different piece of text with NLTK.
# pip install nltk
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = """
Maria Chen founded Bright Path Robotics in Boston in 2022.
The company builds small robots for local schools.
"""

tokens = nltk.word_tokenize(sentence)
print(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)

# The output shows tokenized words, part-of-speech tags, and named entities.
