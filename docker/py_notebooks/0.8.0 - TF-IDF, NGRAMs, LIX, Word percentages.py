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
import sys

# !{sys.executable} -m pip install nltk pyarrow openpyxl plotly

# %%
#import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# %%
from westac_analysis import corpus_fast_tokenizer as cft
import pandas as pd
corpus_version = '0.8.0'
SPEECH_INDEX = pd.read_feather(f'./output/{corpus_version}/speech_index_{corpus_version}.feather')
SPEECH_INDEX = cft.FastCorpusTokenizer.prepare_dataframe(SPEECH_INDEX)
output_path = f'./output/{corpus_version}/'
temporary_path = f'./output/{corpus_version}/temp/'

# %%
# %%time
import importlib
importlib.reload(cft)  
from westac_analysis import corpus_fast_tokenizer as cft

my_cft = cft.FastCorpusTokenizer(SPEECH_INDEX,threads = 24, stop_word_file='ännu fler stoppord till fraser.xlsx', minimum_ngram_count=10)

# %%
##Sanity check, should be 0!
import numpy as np
np.sum(np.asarray(np.sum(my_cft.SPARSE_COUNT,axis=0)).flatten() < my_cft.MINIMUM_NGRAM_COUNT)

# %%
# %%time
from numba import jit
from scipy.sparse import lil_matrix
from multiprocessing import Pool
@jit(nopython=True)
def _calc_ttr(res, vec_words):
    
    for w in vec_words:
        res[w] += 1
    unique = np.sum(res > 0)
    count = np.sum(res)
    # TTR = (number of unique words / total number of words) x 100
    if count > 0:
        return  (unique / count) * 100
    else:
        return -1

# 

def worker_fun(chunk):
    texts = my_cft.VECTORIZED_TEX_DF.vectorized_text.values
    word_count = my_cft.word_count
    ttr_results = np.zeros(len(chunk), dtype=np.float64)
    res = np.zeros(word_count, dtype=np.uint16)
    for i, idx in enumerate(chunk):   
        ttr_results[i] = _calc_ttr(res, texts[idx])
        res[:] = 0
        
    return ttr_results
        

def ttr_function(threads=24):
    chunks = np.array_split(range(len(my_cft.VECTORIZED_TEX_DF)), threads)
    with Pool(threads) as pool:
        results = pool.map(worker_fun, chunks)
    return np.concatenate(results)
    
ttr = ttr_function()
    


# %%
ttr

# %%
SPEECH_INDEX.iloc[123456].u_id

# %%
# TF-IDF på endast 7-grams som börjar på ordet "att"
# Ordfrekvens: Räkna ut vilka ord är de mest vanliga, använd den listan för att sedan se hur många ord som behövs för att täcka in 10, 50, 80 och 90% av texten (tror jag ni menar...)
# Långa ord: För varje "grupp" (decade/party) kolla hur många procent är (5,10,15,20,25)+ bokstäver
# Lix: enligt formel, grupp som ovan

# %%
# %%time
from tqdm.auto import tqdm
import importlib
from westac_analysis import tf_idf_calculator as tfidf
importlib.reload(tfidf)  
from westac_analysis import tf_idf_calculator as tfidf


tfidf_calc = tfidf.TF_IDF_Calculator(my_cft)
word_index = my_cft.CONVERT_DICT['att']
start, end = my_cft.WORD_TO_NGRAM_INDEXES[word_index,:]

# SKip all but those starting with 'att'
data = tfidf_calc.calculate_ngram_groups(['party_abbrev','decade'], ngram_allowed_indexes=(start,end))

# %%
df_dict = tfidf_calc.group_data_to_dataframe(data)

# %%
# # %%time
# from multiprocessing import Pool
# def group_data_to_dataframe(my_tfidfcalc, group_data,no_to_return:int=1000, threads=24):
#     """Converts the output of calculate_ngram_groups to a dataframe


#     """
#     df_dict = {}
    
#     name_keys = list(group_data.keys())
    
#     global worker_fun
    
#     def worker_fun(chunk):
#         worker_keys = [name_keys[x] for x in chunk]
#         worker_results = []
#         for name in worker_keys:
#             group = group_data[name]
#             group_results = []
#             tfidf = group['tfidf']
#             ngram_count = group['ngram_count']

#             sorted_order = np.argsort(tfidf)[::-1][:no_to_return] # Top n ngrams

#             for i in sorted_order:
#                 ngram = my_tfidfcalc.TOKENIZER.ALLOWED_NGRAMS[i,:]
#                 ngram_str = ' '.join([my_tfidfcalc.TOKENIZER.INDEX_TO_WORD[x] for x in ngram])
#                 usages = my_tfidfcalc.get_usage_for_ngram(i)
#                 comb_str = ''
#                 for row, val in usages.sort_values(by='year')[['party_abbrev','year','decade']].groupby('decade').party_abbrev.agg(lambda x: np.unique(x)).items():
#                     comb_str = comb_str + f'{row}: {", ".join(val)}' +' '
#                 group_results.append({'n_gram':ngram_str, 
#                                       'TF-IDF':tfidf[i],
#                                       'used_by':comb_str,                                      
#                                       'Instances in group':ngram_count[i],
#                                       'Instances in any document: ':np.sum(my_tfidfcalc.SPARSE_COUNT_TRANSPOSED.data[i]),
#                                       'ngram_index':i})
#                 # What to do with the usages?
                
#                 df_dict[name] = group_results
#         return df_dict
#     chunks = np.array_split(range(len(name_keys)),threads)
#     with Pool(threads) as pool:
#         results = pool.map(worker_fun, chunks)
#     del worker_fun
#     merged_results = {}
#     for d in results:
#         merged_results = merged_results | d

#     return merged_results
# df_dict2 = group_data_to_dataframe(tfidf_calc, data)

# %%

# %%
# %%time
tfidf_calc.save_ngrams_to_excel(df_dict, output_path)

# %%
# %%time
# # Ordfrekvens: Räkna ut vilka ord är de mest vanliga, använd den listan för att sedan se hur många ord som behövs för att täcka in 10, 50, 80 och 90% av texten (tror jag ni menar...)
most_used = my_cft.get_most_used_words_per_group(['party_abbrev','decade'])
group_cumulative_sums = my_cft.word_use_cumulative_count(most_used)
my_cft.save_cumulative_sums_to_excel(group_cumulative_sums, output_path)

# %%
# %%time
# Långa ord: För varje "grupp" (decade/party) kolla hur många procent är (5,10,15,20,25)+ bokstäver
group_counts = my_cft.count_words_per_group(['party_abbrev','decade'])
group_word_lengths = my_cft.count_group_word_lengths(group_counts)
my_cft.group_word_length_to_excel(group_word_lengths, output_directory=output_path)


# %%
# %%time
# Lix: enligt formel, grupp som ovan
# LIX = (O/M) + ((L * 100)/O)
# O = antal ord i texten
# M = antal meningar i texten
# L = antal långa ord (över 6 bokstäver långa)


lix_per_document, invalid_documents = my_cft.calculate_lix(threads=24)
SPEECH_INDEX['lix'] = lix_per_document
lix_res = []
for (name, year), group in SPEECH_INDEX.groupby(['party_abbrev','decade']):
    lix_res.append({'Party':name, 'Decade':year, 'LIX (mean)':group.lix.mean()})
df = pd.DataFrame(lix_res)
df.to_excel(output_path+'/lix_values.xlsx')

# %%
df2 = df.groupby('Decade').mean().reset_index()
df2['Party'] = 'Mean of all'
df2
import plotly.express as px
px.line(pd.concat([df,df2]),x='Decade',y='LIX (mean)',color='Party',height=600)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
