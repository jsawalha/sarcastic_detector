import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import cleaning_functions as cf
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import bs4 as BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize, WhitespaceTokenizer, wordpunct_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

#Import stopword list from NLTK
# stopword_list=nltk.corpus.stopwords.words('english')


# Import dataset
df = pd.read_csv('../Sarcasm_detector/train-balanced-sarcasm.csv')

#Print columns
print(df.columns)

# First look at the target labels to see if data is balanced
print("Examining target label balance...")
print(df['label'].value_counts())
#Dataset is indeed balanced

#Next, we want see the distribution of character length in sarcastic and non-sarcastic comments
print("Examining distribution of character length in sarcastic and non-sarcastic comments...")
print(df.groupby('label')['comment'].apply(lambda x: x.str.split().str.len().mean()))
    #label
    #0    10.591973
    #1    10.330915

# In my research, I noticed that one row has 10000 characters, so I need to remove it, because it's just spam
#Create new column with length of str in each row
df['comment_length'] = df['comment'].str.len()
# 1010826 rows

print(df.shape)

print("Printing plot of comment length distrubtion across both classes")
g = sns.histplot(data = df[df['comment_length'] < 400], x = 'comment_length', hue = 'label')
plt.legend(title='Class', loc='upper right', labels=['Not sarcastic', 'Sarcastic'])
plt.title('Comment length distribution across both classes')
plt.show()

#plot the top 10 most used sub reddits

sub_df = df.groupby('subreddit')['label'].agg([np.size, np.mean, np.sum])

plt.figure()
ax = sub_df[sub_df['size'] > 1000]['sum'].sort_values(ascending=False).head(10).plot.bar()
plt.title("Top 10 subreddits containing comments")
plt.tight_layout()

#Cleaning data

#Removing nan values
df['comment'].isnull().values.any()
nan_idx = np.where(df['comment'].isnull())[0]
df.drop(nan_idx, inplace=True)

#Clean data
# df['comment'] = df['comment'].str.strip()
df['comment'] = df['comment'].apply(cf.remove_between_square_brackets)
# df['comment'] = df['comment'].apply(denoise_text)
# Not sure we need this one?
# df['comment'] = df['comment'].apply(cf.rem_special_char)
df['comment'] = df['comment'].apply(cf.lower_case)
df['comment'] = df['comment'].apply(cf.stemmer)
df['comment'] = df['comment'].apply(cf.lemmo)

#splitting into train-test-split
print("splitting into train, test split....")
train_x, test_x, y_train, y_test = train_test_split(df['comment'], df['label'], random_state=24)

#translate to numpy variables
train_x = train_x.to_numpy()
test_x = test_x.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#Do a version of the entire dataset for training our final model
X = df['comment'].to_numpy()
y = df['label'].to_numpy()


#Saving files
print("Saving files to numpy...")
np.save("../Sarcasm_detector/clean_data_for_ML/train_x.npy", train_x)
np.save("../Sarcasm_detector/clean_data_for_ML/test_x.npy", test_x)
np.save("../Sarcasm_detector/clean_data_for_ML/y_train.npy", y_train)
np.save("../Sarcasm_detector/clean_data_for_ML/y_test.npy", y_test)

np.save("../Sarcasm_detector/clean_data_for_ML/X_ovr.npy", X)
np.save("../Sarcasm_detector/clean_data_for_ML/y_ovr.npy", y)







# #TDIDF vectorizer
# tv = TfidfVectorizer(min_df=0, max_df=0.25, use_idf=True, ngram_range=(1,3), lowercase=False)

# tfidf_train_x = tv.fit_transform(train_x)
# tfidf_test_x = tv.transform(test_x)













