sentence_case_documents = ['Hi, HELLO, how Are You you???', 'Am FINE, Thank YOU you!!!']

# Convert to Lowercase
lower_case_documents = []

for i in sentence_case_documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

# Remove Punctuation

import string

sans_punctuation_documents = []
translator = str.maketrans('', '', string.punctuation)
for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(translator))
print(sans_punctuation_documents)

# Tokenize
preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

# Count Frequencies
frequency_list = []

from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
print(frequency_list)

########## No using Scikit Learn - Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(stop_words='english')  # remove stop_words to see the effect
print(count_vector)

count_vector.fit(sentence_case_documents)
print(count_vector.get_feature_names())
doc_array = count_vector.transform(sentence_case_documents).toarray()
print(doc_array)

import pandas as pd

frequency_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())
print(frequency_matrix)
