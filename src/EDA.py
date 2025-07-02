# 📦 Step 0: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud

# 📁 Step 1: Load the dataset
df = pd.read_csv("Data/complaints.csv", low_memory=False)
print("Dataset shape:", df.shape)
df.head()

# Columns overview
df.columns

# Count of missing narratives
df['Consumer complaint narrative'].isnull().sum()

# Drop complaints with missing narratives
df = df.dropna(subset=['Consumer complaint narrative'])

# Word count analysis
df['word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(x.split()))

# Visualize word counts
sns.histplot(df['word_count'], bins=50, kde=True)
plt.title("Distribution of Complaint Narrative Length")
plt.xlabel("Word Count")
plt.ylabel("Number of Complaints")
plt.show()


relevant_products = [
    "Credit card", 
    "Personal loan", 
    "Buy Now, Pay Later (BNPL)", 
    "Savings account", 
    "Money transfer, virtual currency, or money service"
]

# Some datasets may not have exact match for "BNPL", you may need to adjust
df = df[df['Product'].isin(relevant_products)]
print("Filtered dataset shape:", df.shape)
df['Product'].value_counts().plot(kind='barh', title="Complaints by Product")
plt.show()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)


df[['cleaned_narrative', 'Product', 'Company', 'word_count']].to_csv("Data/filtered_complaints.csv", index=False)
print("Saved cleaned data to Data/filtered_complaints.csv")
