#!/usr/bin/env python
# coding: utf-8

# # ****Tracing Ottoman Emperors Through Text Analytics: A Historical Expedition Using Natural Language Processing (NLP)

# ![image-2.png](attachment:image-2.png)

# ### Introduction:
# The "NLP-Driven Exploration of Ottoman Trade Route" is a comprehensive endeavor aimed at unearthing historical insights related to the Ottoman trade routes using advanced Natural Language Processing (NLP) techniques. The primary data source for this project is the esteemed book, "The Ottoman Turks and the Routes of Oriental Trade." Our goal is to extract and analyze names associated with historical contexts, thereby shedding light on the remarkable trade routes of the Ottoman Empire.
# 
# ### Project Objectives:
# 
#  - **Textual Analysis:** The project starts with in-depth textual analysis of the book, focusing on the descriptions and narratives that refer to historical figures, locations, and trade routes.
# 
#  - **Entity Recognition:** Using NLP tools and techniques, we identify and extract entities within the text, particularly focusing on names (e.g., people, places) relevant to Ottoman trade history.
# 
#  - **Relation Extraction:** The project explores relationships among entities by determining how they are connected within the context of the Ottoman trade routes.
# 
#  - **Visual Representation:** We create Wordcloud graphs and data visualizations to illustrate the connections and relationships among historical figures and trade locations

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import spacy
import nltk
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from nltk.stem import PorterStemmer
from collections import Counter
from spacy import displacy
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


# In[35]:


with open("ottoman.txt", 'r', encoding='latin1') as f:
    text = f.read()
    print(text)


# In[36]:


chars = sorted(set(text))
print(chars)


# In[37]:


#Using Character Level Tokenizer
  
string_to_int = {char: i for i,char in enumerate(chars)}
int_to_string = {i: char for i,char in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

encoded_text = encode(text)
decoded_text = decode(encoded_text)

print(encoded_text)
print(decoded_text)


# In[38]:


data = torch.tensor(encode(text), dtype=torch.long)
print(data[:1000])


# In[39]:


# Validation and Training Split


# In[40]:


n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]


# In[41]:


block_size = 200

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('when input is', context, 'target is', target)


# In[42]:


nlp_core_web_sm = spacy.load('en_core_web_sm')

# Process the text with the loaded spaCy model
doc = nlp_core_web_sm(text)   
# Print the pipeline components of the 'en_core_web_sm' model
print(nlp_core_web_sm.pipeline)

for token in doc:
    print(token)


# In[43]:


for token in doc:
    print(token.text, token.pos_, token.dep_)


# In[44]:


for token in doc:
    print(token, " *|* ", token.pos_, " *|* ", token.lemma_)


# In[45]:


for ent in doc.ents:
    print(ent.text, " *|* ", ent.label_, " *|* ", spacy.explain(ent.label_))
    


# In[46]:


displacy.render(doc, style='ent')


# In[47]:


stemmer = PorterStemmer()


# In[48]:


word_list = []

for review_text in doc:
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
            
word_freq = Counter(word_list)

for index, number in enumerate(word_freq.items()):
    print(index, number)
    
print(word_freq, end="\n")


# In[49]:


# Get the top 50 words and their frequencies
top_word_freq = word_freq.most_common(100)

# Separate the words and frequencies for plotting
top_words, top_frequencies = zip(*top_word_freq)

# Plot word frequencies for the top 50 words
plt.figure(figsize=(14, 8))
plt.bar(top_words, top_frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 50 Word Frequencies in Reviews')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[50]:


for token in doc:
    print(token, "|", token.pos_, "*****", spacy.explain(token.pos_), "***Token Tag: ", token.tag_, " *|* ", spacy.explain(token.tag_))


# In[51]:


for token in doc:
    if token.pos_ in ["NOUN", "PROPN"]:
        print(token, " | ", token.pos_, " | ", spacy.explain(token.pos_))


# In[52]:


# Get common nouns and proper nouns
nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
word_freq = Counter(nouns)

top_50_nouns = word_freq.most_common(50)

# Separate the words and frequencies for plotting
all_words, all_frequencies = zip(*top_50_nouns)


plt.figure(figsize=(12, 6))
sns.barplot(x=list(all_words), y=list(all_frequencies), palette="viridis")
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 50 Noun Frequencies')
plt.xticks(rotation=90)  


# In[53]:


# distribution curve using the frequencies
mu, std = np.mean(list(all_frequencies)), np.std(list(all_frequencies))
x = np.linspace(0, max(list(all_frequencies)), 100)
p = norm.pdf(x, mu, std)
plt.twinx()  
plt.plot(x, p, color='red', linestyle='dotted', label='Distribution Curve')
plt.ylabel('Distribution')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# In[54]:


filtered_tokens = []

for token in doc:
    if token.pos_ not in ["SPACE", "X", "PUNCT"]:
        filtered_tokens.append(token)

print(filtered_tokens)


# In[ ]:





# In[55]:


for ent in doc.ents:
    print(ent.text, "|", ent.label_, spacy.explain(ent.label_))


# In[56]:


# Extract entities and their types
entity_data = [(ent.text, ent.label_) for ent in doc.ents]

# Count the frequency of each entity type
entity_type_counts = Counter(entity_data)

# Extract entity types and their corresponding counts
types = [entity_type for entity_type, _ in entity_type_counts.keys()]
counts = list(entity_type_counts.values())

types, count = zip(*entity_type_counts.items())
types, count = zip(*sorted(zip(types, counts), key=lambda x: x[1], reverse=True))

mean_count = np.mean(counts)
median_count = np.median(counts)
std_deviation = np.std(counts)


print("Mean Count:", mean_count)
print("Median Count:", median_count)
print("Standard Deviation:", std_deviation)


# In[57]:


words = doc.text.lower().split()
vocabulary = list(set(words))

words


# In[58]:


word_to_index = {word: i for i, word in enumerate(vocabulary)}
print(word_to_index)


# In[59]:


def one_hot_encoding(word, word_to_index):
    one_hot_vector = np.zeros(len(word_to_index))
    
    if word in word_to_index:
        index = word_to_index[word]
        one_hot_vector[index] = 1
        
    return one_hot_vector

encoded_text = [one_hot_encoding(word, word_to_index) for word in words]


print("Text:", text)


# In[60]:


print("\nWord to Index Mapping:", word_to_index)


# In[61]:


print("\nOne-Hot Encoded Text:")
for word, encoding in zip(words, encoded_text):
    print(f"{word}: {encoding}") 


# In[62]:


entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_texts, entity_labels = zip(* entities)

label_encoder = LabelEncoder()

encoded_entity_labels = label_encoder.fit_transform(entity_labels)

for entity, label, encoded_label in zip(entity_texts, entity_labels, encoded_entity_labels):
    print(f"Entity: {entity}\tLabel: {label}\tEncoded Label: {encoded_label}")
    


# In[63]:


# Extract entities and their types
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Encode entity labels using LabelEncoder
entity_texts, entity_labels = zip(*entities)
label_encoder = LabelEncoder()
encoded_entity_labels = label_encoder.fit_transform(entity_labels)


entity_label_mapping = {entity: encoded_label for entity, encoded_label in zip(entity_texts, encoded_entity_labels)}
person_entities = [entity for entity, label in zip(entity_texts, entity_labels) if label == 'PERSON']
person_entity_counts = Counter(person_entities)



# In[64]:


top_50_person_entities = person_entity_counts.most_common(50)

person_entity_texts, person_entity_counts = zip(*top_50_person_entities)

plt.figure(figsize=(12, 6))
plt.bar(person_entity_texts, person_entity_counts)
plt.xlabel("PERSON Entity")
plt.ylabel("Frequency")
plt.title("Top 50 PERSON Entity Frequency Plot")
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()


# In[65]:


# Count the frequency of each PERSON entity
person_entity_counts = Counter(person_entities)

person_entity_counts


# In[66]:


# Join the PERSON entities into a single string separated by spaces
person_entities_text = " ".join(person_entities)

wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis")

wordcloud.generate(person_entities_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of PERSON Entities")
plt.show()


# In[ ]:


# Going to Extract the Countries Names and Analyzing it's frequency with Wordcloud


# In[73]:


countries_entity = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"]
countries_entity_counts = Counter(countries_entity)
print(countries_entity_counts)


# In[79]:


# Extract entities recognized as countries ("GPE")
country_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
country_entity_counts = Counter(country_entities)


country_names = list(country_entity_counts.keys())
counts = list(country_entity_counts.values())


plt.figure(figsize=(12, 8))
sns.barplot(x=counts, y=country_names, orient="h")
plt.xlabel("Frequency")
plt.ylabel("Country Name")
plt.title("Country Name Frequencies in the Text")
plt.xticks(rotation=90)
plt.show()


# # Machine Learning Phase

# In[127]:


with open("ottoman.txt", "r", encoding="utf-8") as file:
    text = file.read().splitlines()
    
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(text)

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)

cluster_labels = kmeans.labels_

for i, label in enumerate(cluster_labels):
    print(f"Document {i} is in Cluster {label}")


# ```
# Essentially, this code is grouping similar documents (lines of text) from the "ottoman.txt" file into clusters based on their content using K-Means clustering. It's a way to identify patterns and similarities within the text data without any prior labeling or categorization. 
# ```

# In[128]:


X_train, X_test, y_train, y_test = train_test_split(text, cluster_labels, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[129]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

y_pred = knn.predict(X_test_tfidf)


# In[130]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[132]:


# ROC AUC Curve (Only applicable for binary classification)

if len(set(y_test)) == 2:
    y_scores = knn.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)
    
    
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


# In[136]:


param_grids = {
    'n_neighbors': [3,5,7,9],
    'weights': ['uniform', 'distance'],
    'p': [1,2],  # 1 is manhattan distance and 2 is euclidean distance
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grids, cv=5) # 5-fold cross-validation

grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_
best_knn = grid_search.best_estimator_

y_pred = best_knn.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)


# In[138]:


# Create a pipeline with TfidfVectorizer and multiple classifiers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', XGBClassifier()),
    # You can replace this with other classifiers
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)


# In[143]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=6, scoring='roc_auc')

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)


roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc}")

print("Best Hyperparameters:")
print(grid_search.best_params_)


# ```
# Specifically, it utilizes the Grid Search with Cross-Validation (GridSearchCV) technique to systematically explore and identify the most effective hyperparameters for a classification model, with a focus on an XGBoost classifier. The 'param_grid' dictionary defines a range of hyperparameter values to be considered. This GridSearchCV instance is configured for 5-fold cross-validation, which divides the training dataset into five subsets, allowing the model to be trained and evaluated multiple times.
# 
# The key objective of this process is to pinpoint the hyperparameters that produce the highest ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) score. The ROC-AUC score is a widely-used metric for assessing binary classifiers, providing insights into the model's ability to distinguish between positive and negative classes. The results are then printed out, including the best hyperparameters and the ROC-AUC score. Ultimately, this method streamlines the intricate task of hyperparameter optimization, enhancing the model's predictive capabilities and generalizability to unseen data.
# 
# ```

# In[145]:


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_score = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ```
# This code segment focuses on evaluating the performance of a binary classification model. It starts by selecting the best-performing model from a hyperparameter tuning process using GridSearchCV. Once the best model is identified, it is trained on the training data to prepare it for predictions. To assess the model's predictive accuracy, it computes the predicted probabilities for the positive class (class 1) in the test dataset.
# 
# The critical part of this code is the generation of a Receiver Operating Characteristic (ROC) curve and the calculation of the Area Under the Curve (AUC). The ROC curve is a graphical representation of the trade-off between the true positive rate (TPR) and the false positive rate (FPR) at different probability thresholds. A model's performance can be evaluated by examining how well it separates positive and negative instances. The AUC score quantifies this separation, with a higher AUC indicating better model performance.
# ```

# In[ ]:




