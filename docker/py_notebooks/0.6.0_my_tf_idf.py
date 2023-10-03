# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
corpus_version = '0.7.0'
# SPEECH_INDEX = pd.read_feather('speech_index_0.4.6_dev.feather')
SPEECH_INDEX = pd.read_feather(f'./output/{corpus_version}/speech_index_{corpus_version}.feather')
# SPEECH_INDEX = pd.read_feather(f'speech_index_{corpus_version}.feather')

# %%
SPEECH_INDEX['decennium'] = SPEECH_INDEX.year.apply(lambda x: int(x/10)*10)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
# import swifter
from tqdm.auto import tqdm

group_texts = {}
for name, group in tqdm(SPEECH_INDEX.groupby(['party_abbrev','decennium'])):
    group_texts[name] = ' '.join([' '.join([str(x).lower() for x in y]) for y in group.text.values])

# %%
# HERE WE SPECIFY THE NGRAM SIZE
our_ngram_range = (10,10)

# %%
##----
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
import functools
from collections import Counter
# from westac_analysis.term_frequency import _do_counter
vectorizer = CountVectorizer(lowercase=True, ngram_range = our_ngram_range)
analyzer = vectorizer.build_analyzer()
counter = Counter()    

for text in tqdm(group_texts.values()):
    counter.update(analyzer(text))

# %%
relevant_ngrams = {k:v for k, v in counter.items() if (v > 10)}
len(relevant_ngrams)

# %%
relevant_ngram_vocabolary = set(relevant_ngrams.keys())

# %%
df = pd.read_excel('ännu fler stoppord till fraser.xlsx',header=None)
df.columns = ['Stoppord']
df

# %%
df.Stoppord.str.lower().values

# %%
remove_list = []
words_to_remove = df.Stoppord.str.lower().values
for ngram in list(relevant_ngram_vocabolary):
    for word in words_to_remove:
        if word.endswith('*'):
            matching = any([x.startswith(word[:-1]) for x in ngram.split()])
        elif word.startswith('*'):
            matching = any([x.endswith(word[1:]) for x in ngram.split()])
        else:
            matching = any([x == word for x in ngram.split()])
        if matching:
            remove_list.append(ngram)
            break


# %%
print(f'Remove list of length: {len(remove_list)}')
print(f'Relevant ngrams of length: {len(relevant_ngram_vocabolary)}')
b = relevant_ngram_vocabolary - set(remove_list)
print(f'Remaining ngrams of length: {len(b)}')
relevant_ngram_vocabolary = b

# %%
for x in relevant_ngram_vocabolary:
    if 'talman' in x:
        print(x)
        
    

# %%
ngram_to_index = {ngram:idx for idx, ngram in enumerate(relevant_ngram_vocabolary)}
index_to_ngram = {v:k for k,v in ngram_to_index.items()}

# %%
import scipy.sparse as sparse
import numpy as np
mat = sparse.lil_matrix((len(group_texts), len(relevant_ngram_vocabolary)),dtype=np.int32)
mat                       

# %%
for row, text in tqdm(enumerate(group_texts.values()))                        :
    for ngram in analyzer(text):
        col = ngram_to_index.get(ngram,None)
        if col is not None:
            mat[row,col] += 1

# %%
np.array([[1,2,3],[3,4,5],[6,7,8]]) * (1 / np.array([0.5,0.5,0.5]))

# %%
document_frequency = np.sum(mat > 0,axis = 0)
document_frequency_mod = (1 / document_frequency).squeeze()
print(document_frequency_mod.shape)
print(np.max(document_frequency_mod))
print(np.min(document_frequency_mod))

# %% [markdown]
# # Multiply raw ngram count with with IDF to get TF-IDF

# %%
tfidf_mat = mat.multiply(document_frequency_mod)
tfidf_mat = tfidf_mat.tolil()

# %%
# df = SPEECH_INDEX
# for idx in range(len(df)):
#     row = df.iloc[idx]
#     for text in row.text:
#         if 'var femte svensk' in text.lower():
#             print(text)
#             print('----------')
            
    

# %%
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages')
import swifter

# %%
word_count_df = pd.DataFrame([{'Name':name} for name in group_texts.keys()])
word_count_analyzer = CountVectorizer(lowercase=True, ngram_range = (1,1)).build_analyzer()
word_count_df['word_count'] = word_count_df.Name.swifter.apply(lambda x: len(word_count_analyzer(group_texts[x])))
word_count_df

# %%
# %%time
party_index = [party for party,year in group_texts.keys()]
tfid_mat_has_entry = (tfidf_mat > 0).T.tolil()

keys = list(group_texts.keys()) 
def parties_using_ngram(ngram_index):
    l =  sorted([keys[x] for x in tfid_mat_has_entry.rows[ngram_index]], key = lambda x: x[1])
    return ', '.join([f'{decade} - ({", ".join([party for party, year in l if year == decade])})' for decade in sorted(np.unique([x[1] for x in l]))])


def get_ordered_appearance(ngram_index):
    a = [party for party, year in sorted([keys[x] for x in tfid_mat_has_entry.rows[ngram_index]], key = lambda x: x[1])]
    u, ind = np.unique(a, return_index=True)
    return ', '.join(u[np.argsort(ind)])
    
# parties_using_ngram(ngram_to_index['var femte svensk arbetsför ålder står utanför'])

from pathlib import Path
Path(f'./output/{corpus_version}/results/tf-idf_{our_ngram_range}').mkdir(exist_ok=True)

# Path("/my/pythondirectory").mkdir(parents=True, exist_ok=True)

for row, ((name, year), text) in tqdm(enumerate(group_texts.items())):
    # if year != 1980 or name != 'M':
    #     continue
    # print(f'{row}: {year} - {name}')
    word_count = word_count_df[word_count_df.Name == (name, year)].word_count.values[0]
    tfidf_row = tfidf_mat.getrow(row).toarray().squeeze()
    top_indexes = (np.argsort(tfidf_row)[::-1][:1000])
    df = pd.DataFrame([{'ngram':index_to_ngram[idx],
                        'TF-IDF':tfidf_row[idx],
                        'TF':mat[row,idx], 
                        'DF':document_frequency[0,idx],
                        'Total count':counter[index_to_ngram[idx]],
                        'Parties using ngram':get_ordered_appearance(idx),
                        'Parties using ngram (long)':parties_using_ngram(idx),
                       } for idx in top_indexes])
    # df.ngram.apply(lambda x: parties_using_ngram(ngram_to_index[x])[0])
    # df['Parties using ngram'] = [get_ordered_appearance(x) for x in top_indexes]
    df['TF-IDF (corrected)'] = df['TF-IDF'].apply(lambda x: x/word_count)
    df.insert(2, 'TF-IDF (corrected)', df.pop('TF-IDF (corrected)'))
    # df = df.sort_values(by='TF-IDF (corrected)',ascending=False)
    df.to_excel(f'./output/{corpus_version}/results/tf-idf_{our_ngram_range}/{year}_{name}.xlsx')
    
    # display(df.head(10))


# %%

# %%
class WordCounter:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS word_count (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT,
    UNIQUE (word)
);''')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.cursor.close()
            self.conn.close()
            
            
    def do_sql(self, sql_statement):
        try:
            self.cursor.execute(sql_statement)
            return self.cursor.fetchall()
        except sqlite3.IntegrityError:
            pass     
            
            
    def insert_word(self, word):
        try:
            self.cursor.execute("INSERT OR IGNORE INTO word_count (word) VALUES (?)", (word,))
        except sqlite3.IntegrityError:
            pass            
        
    def query_word(self, word):
        self.cursor.execute("SELECT id FROM word_count WHERE word=?", (word,))
        result = self.cursor.fetchone()
        return result[0]

    def get_all_words(self):
        self.cursor.execute("SELECT word FROM word_count ORDER BY id")
        return [x[0] for x in self.cursor.fetchall()]

# %%
    
from tqdm.auto import tqdm

group_texts = {}
for name, group in tqdm(SPEECH_INDEX.groupby(['party_abbrev','decennium'])):
    group_texts[name] = ' '.join([' '.join([str(x).lower() for x in y]) for y in group.text.values])
    
import sqlite3
import numpy as np

group_data = {}

# # !rm speech_index_db.db

# with WordCounter('speech_index_db.db') as wc:
    # wc.insert_word('hello')    
    # wc.insert_word('hello2')    
    # wc.insert_word('hello6')    
    # wc.insert_word('hello6')    
    # wc.insert_word('hello6')    
    # wc.insert_word('hello3')    
    # wc.insert_word('hello6')    
    # wc.insert_word('hello6')    
    # print(wc.get_all_words())

for name, text in tqdm(group_texts.items()):
    with WordCounter('speech_index_db.db') as wc:
        # print(len(np.unique(text.split())))
        vectorizer = CountVectorizer()
        tokenized_string = vectorizer.build_analyzer()(text)
        for word in tqdm(tokenized_string):
            wc.insert_word(word)

# %%
with WordCounter('speech_index_db.db') as wc:
    
    wc.do_sql('''CREATE TABLE IF NOT EXISTS word_count2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT,
        UNIQUE (word)
    );''')
    wc.do_sql('INSERT INTO word_count2 (word) SELECT word FROM word_count')
    wc.do_sql('UPDATE word_count2 SET id = id - 1;')    

# %%
with WordCounter('speech_index_db.db') as wc:
    res = wc.do_sql('SELECT * from word_count2 WHERE word = "test"')
    
res    

# %%
with WordCounter('speech_index_db.db') as wc:
    res = wc.do_sql('SELECT COUNT(*) from word_count2')
    row_count = res[0][0]
    
row_count    

with WordCounter('speech_index_db.db') as wc:
    res = wc.do_sql('SELECT * from word_count2 LIMIT 10')

# %%
group_int_texts = {}
with WordCounter('speech_index_db.db') as wc:
    for name, text in tqdm(group_texts.items()):    
        # print(len(np.unique(text.split())))
        vectorizer = CountVectorizer()
        tokenized_string = vectorizer.build_analyzer()(text)
        group_int_texts[name] = np.array([wc.query_word(word) for word in tqdm(tokenized_string)],dtype=np.int32)


# %%
group_int_texts[('M', 1960)].shape

# %%
with WordCounter('speech_index_db.db') as wc:
    print(type(wc.query_word('dom')))
