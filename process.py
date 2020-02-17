import re
from string import digits

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

remove_digits = str.maketrans('', '', digits)

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# Очистка данных

def dataframe_cleaner(df):
    for index, row in df.iterrows():
        # Чистим ссылки
        text = re.sub(r'(http|https)?:\/\/.*[\r\n]*', '', row['text'], flags=re.MULTILINE)
        # Обращения к юзерам
        text = re.sub(r"@\w+", '', text, flags=re.MULTILINE)
        # Все спец символы
        text = re.sub(r'\W+', ' ', text)

        text = text.lower().replace('#', '').replace(' via ', '')
        text = text.translate(remove_digits)
        df.at[index, 'text'] = text


dataframe_cleaner(train_df)


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf = text_clf.fit(train_df.text, train_df.target)

dataframe_cleaner(test_df)

predicted = text_clf.predict(test_df.text)
ids = test_df['id'].to_list()
result = pd.DataFrame(data=dict(
    id=ids,
    target=predicted
))
result.to_csv('subs.csv', index=False)