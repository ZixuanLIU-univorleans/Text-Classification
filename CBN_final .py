import pandas as pd
from nltk.tokenize import word_tokenize
data = pd.read_excel('DATA_Bulgari.xlsx')
data['Feedback - Comment - English (EN)'] = data['Feedback - Comment - English (EN)'].str.lower()
#### stop words  ####
stopwords = [line.strip() for line in open('english.txt', 'r', encoding='utf-8').readlines()]
result = []
for text in data['Feedback - Comment - English (EN)'].astype(str):
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords]
    result.append(','.join(words))

data['tokens_comment'] = result
data.to_excel('new_data.xlsx', index=False)

####  TF-IDF  ####
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer()
# fit_transform() pour obtenir TF-IDF matrix
tfidf_matrix = tfidf_model.fit_transform(data['tokens_comment'])
tfidf_array = tfidf_matrix.toarray()
#print(tfidf_array)

words_set = tfidf_model.get_feature_names_out()
#print(words_set)

# dataframe to show the TF-IDF scores
df_tf_idf = pd.DataFrame(tfidf_array, columns=words_set)
#print(df_tf_idf)

####  test_size  ####
X = data['tokens_comment']
y = data['Sentiment']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

test = pd.DataFrame(tfidf_model.fit_transform(X_train).toarray(), columns=tfidf_model.get_feature_names_out())
#print(test)

####  CNB  ####
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

X_train_vect = tfidf_model.fit_transform(X_train)
nb.fit(X_train_vect, y_train)
train_score = nb.score(X_train_vect, y_train)
#print(train_score)

X_test_vect = tfidf_model.transform(X_test)
#print(X_test, y_test,nb.predict(X_test_vect))
#print(nb.score(X_test_vect, y_test))


nb_result = nb.predict(X_test_vect)
#predict_proba(X)[source]
#data['nb_result'] = nb_result
a = { 'tokens_comment': X_test, 'y_test':y_test,'nb_predict':nb_result}
data1 = pd.DataFrame(a)
data1.to_excel("nb_test.xlsx")
# test = pd.DataFrame(tfidf_model.fit_transform(X).toarray(), columns=tfidf_model.get_feature_names_out())
# print(test.head())


#data.to_excel("nb_result.xlsx",index=False)


from sklearn import metrics
predict_labels = nb.predict(X_test_vect)
F1_score = metrics.f1_score(y_test, predict_labels,pos_label='Positive')
print(F1_score)
