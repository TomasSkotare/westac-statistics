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

# !{sys.executable} -m pip install nltk pyarrow openpyxl

# %%
import pandas as pd
# import tracemalloc
import os
from westac_analysis import corpus_tokenizer as ct

# Start tracing memory usage
# tracemalloc.start()

corpus_version = '0.8.0'
SPEECH_INDEX = pd.read_feather(f'./output/{corpus_version}/speech_index_{corpus_version}.feather')
SPEECH_INDEX['decade'] = SPEECH_INDEX.year.apply(lambda x: x - (x%10))
temporary_path = f'./output/{corpus_version}/temp/'

# %%
import nltk
from collections import Counter

SPEECH_INDEX['text_merged'] = SPEECH_INDEX.text.apply(lambda x: ' '.join(iter(x)).lower())
SPEECH_INDEX.drop(['file_name','alternative_names','text','u_ids'],axis=1,inplace=True)

# %%
# %%time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm.auto import tqdm
from multiprocessing import Pool
import numpy as np
le = LabelEncoder()
transformed_text = []


def __create_dict(texts):
    # all_words = defaultdict(int)
    word_set = set()
    for text in SPEECH_INDEX.text_merged.values[texts]:
        word_set.update((x for x in nltk.regexp_tokenize(text, pattern=R"(?u)\b\w\w+\b")))
    return word_set

def threaded_find_words(threads):
    chunks = np.array_split(range(len(SPEECH_INDEX)),threads)
    with Pool(threads) as pool:
        workers = pool.map_async(__create_dict, chunks)
        all_words = set()
        for d in workers.get():
            all_words.update(d)
    convert_dict = {k:v for k,v in zip(all_words, range(len(all_words)))}    
    return convert_dict

convert_dict = threaded_find_words(threads=24)
index_to_word = [k for k,v in convert_dict.items()]
# for text in tqdm(SPEECH_INDEX.text_merged[:100]):
#     transformed_text.append(le.transform(nltk.regexp_tokenize(text, pattern=R"(?u)\b\w\w+\b")))

# %%

# %%
len(convert_dict)


# %%
# %%time

# def __create_vector(chunk):
#     vectors = []
#     for idx in chunk:
#         uid, text = SPEECH_INDEX.u_id.values[idx], SPEECH_INDEX.text_merged.values[idx]
#         vectors.append((uid, np.array([convert_dict[x] for x in nltk.regexp_tokenize(text, pattern=R"(?u)\b\w\w+\b")],dtype=np.uint32)))
#     return vectors

def __create_vector(chunk):
    vectors = []
    word_count = np.zeros(len(convert_dict))
    for idx in chunk:
        uid, text = SPEECH_INDEX.u_id.values[idx], SPEECH_INDEX.text_merged.values[idx]
        vectorized_text = np.array([convert_dict[x] for x in nltk.regexp_tokenize(text, pattern=R"(?u)\b\w\w+\b")],dtype=np.uint32)
        for i in range(len(vectorized_text)):
            word_count[vectorized_text[i]] += 1
        vectors.append((uid, vectorized_text))
    return (vectors, word_count)

def threaded_vectorization(threads):
    chunks = np.array_split(range(len(SPEECH_INDEX)),threads)
    total_word_counts = []
    with Pool(threads) as pool:
        workers = pool.map_async(__create_vector, chunks)
        all_vectors = []
        for d, word_counts in workers.get():
            all_vectors.extend(d)
            total_word_counts.append(word_counts)
    summed_word_counts = np.sum(np.vstack(total_word_counts),axis=0)
    return all_vectors, summed_word_counts

vectorized_documents, summed_word_counts = threaded_vectorization(24)     

# %%
vectorized_tex_df = pd.DataFrame([{'u_id':x, 'vectorized_text':y} for x, y in vectorized_documents])

# %%
# %%time
from collections import Counter
ngram_length = 7

df = pd.read_excel('ännu fler stoppord till fraser.xlsx',header=None)
df.columns = ['Stoppord']
stop_words = [convert_dict[x] for x in df.Stoppord.values if convert_dict.get(x,None) is not None]
stop_set = set(stop_words) 
uncommon_words = summed_word_counts < 10

# Create a boolean array with all words that should not be used
remove_array = uncommon_words
for word in stop_words:
    remove_array[word] = True
remove_count = np.sum(remove_array)
print(f'Words to remove: {remove_count} ({(remove_count / len(remove_array)) * 100:.2f}%)')


def __count_ngrams(chunk):
    vectors = []
    data = np.concatenate(vectorized_tex_df.vectorized_text.values[chunk])
    has_remove_word = remove_array[data]
    # actually_remove = np.zeros(len(has_remove_word), bool)
    # for i in range(len(has_remove_word)):
    #     # Must remove all ngrams containing the word, by setting the value to True for them.
    #     # We predict the following ngram_length ngrams will contain the word!
    #     if has_remove_word[i]:
    #         actually_remove[i:i+ngram_length] = True
    
    data2d = np.lib.stride_tricks.sliding_window_view(data, window_shape=ngram_length)
    has_remove_word2d = np.lib.stride_tricks.sliding_window_view(has_remove_word, window_shape=ngram_length)
    remove_rows = np.any(has_remove_word2d,axis=1)
    # Remove all rows containing words we do not want to keep
    
    print(f'Remove uncommon or stop words: {((np.sum(remove_rows) / len(remove_rows)) * 100):.2f}%')
    before = data2d.shape
    data2d = data2d[~remove_rows,:]
    print(f'Before: {before} - after: {data2d.shape}')
    unique, count = np.unique(data2d, axis=0, return_counts=True)
    return (unique, count)

def threaded_ngram_counting(threads):
    chunks = np.array_split(range(len(SPEECH_INDEX)),threads)
    with Pool(threads) as pool:
        workers = pool.map_async(__count_ngrams, chunks)
        all_counters = []
        for d in workers.get():
            all_counters.append(d)
    return all_counters

all_counters = threaded_ngram_counting(24)

# %%
# %%time
from numba import jit

@jit(nopython=True)
def is_sorted(a,b):
    for idx in range(len(a)):
        if a[idx] < b[idx]:
            return 1
        if a[idx] > b[idx]:
            return -1
    return 0

@jit(nopython=True)
def merge_sorted_arrays(a,b,ac,bc):
    if a.shape[1] != b.shape[1]:
        print('Error! Shapes not same!')
        return
    merged = np.zeros((len(a)+len(b), a.shape[1]),dtype=np.uint32)
    merged_c = np.zeros(len(a)+len(b),dtype=np.uint32)
    m_idx = 0
    a_idx = 0
    b_idx = 0
    a_max = a.shape[0]
    b_max = b.shape[0]
    while a_idx < a_max and b_idx < b_max:
        res = is_sorted(a[a_idx,:], b[b_idx,:])
        if res == 0:
            # Same row, merge count
            merged_c[m_idx] = ac[a_idx] + bc[b_idx]
            merged[m_idx,:] = a[a_idx,:]            
            a_idx += 1
            b_idx += 1
        elif res == -1:
            # b should be before a
            merged_c[m_idx] = bc[b_idx]
            merged[m_idx,:] = b[b_idx,:]            
            b_idx += 1
        else: 
            # a should be before b
            merged_c[m_idx] = ac[a_idx]
            merged[m_idx,:] = a[a_idx,:]            
            a_idx += 1
        m_idx += 1
    if a_idx < a_max:
        remaining = a_max - a_idx
        merged[m_idx:m_idx+remaining,:] = a[a_idx:,:]
        merged_c[m_idx:m_idx+remaining] = ac[a_idx:]
        m_idx += remaining        
    if b_idx < b_max:
        remaining = b_max - b_idx
        merged[m_idx:m_idx+remaining,:] = b[b_idx:,:]
        merged_c[m_idx:m_idx+remaining] = bc[b_idx:]
        m_idx += remaining        
    return merged[:m_idx,:], merged_c[:m_idx]

# res, res_c = merge_sorted_arrays(arr1,arr2,count1,count2)    


# %%
# %%time
## TODO: This can be done with multiprocessing too, at the cost of memory...
merged_ngram, merged_counter = None, None
while all_counters: 
    ngrams, counter = all_counters.pop()
    if merged_ngram is None:
        merged_ngram = ngrams
        merged_counter = counter
        continue
    merged_ngram, merged_counter = merge_sorted_arrays(merged_ngram, ngrams, merged_counter, counter)

# %%
merged_ngram.shape

# %%
merged_ngram.nbytes / 1024 / 1024 / 1024

# %%
allowed_ngrams = merged_ngram[merged_counter > 10,:]
allowed_counter = merged_counter[merged_counter > 10]

# %%
(allowed_ngrams.shape[0] / merged_ngram.shape[0]) * 100

# %%
csort = np.argsort(allowed_counter)
for idx in csort[::-1][:10]:
    print(allowed_counter[idx], ': ', ' '.join([index_to_word[x] for x in allowed_ngrams[idx,:]]))

# %%
np.sum([x[1].nbytes for x in vectorized_documents]) / 1024/1024


# %%
# %%time 

# @jit(nopython=True)
def allowed_ngrams_per_doc(data, allowed_per_col, ngram_length):
    if len(data) < ngram_length:
        return []
    data2d = np.lib.stride_tricks.sliding_window_view(data, window_shape=ngram_length)
    # print(data2d.shape)
    for col in range(ngram_length):
        data2d = data2d[allowed_per_col[data2d[:,col],col]]
    return data2d


allowed_per_col = np.zeros((len(convert_dict),ngram_length), dtype=bool)
for col in range(allowed_per_col.shape[1]):
    allowed_per_col[allowed_ngrams[:,col],col] = True

# %%
# group_results = []    
# for name, group in SPEECH_INDEX.groupby(['party_abbrev','decade']):
#     print(name)
#     df = vectorized_tex_df[vectorized_tex_df.u_id.isin(group.u_id)]
#     document_texts = df.vectorized_text.values    
#     uids = df.u_id.values
    
#     group_ngrams = []    
#     for idx in range(len(document_texts)):
#         data = document_texts[idx]
#         group_ngrams.append((uids[idx], allowed_ngrams_per_doc(data, allowed_per_col, ngram_length)))
#     group_results.append((name, group_ngrams))
    

# %%
# uid_ngrams = []
# for group_name, ngram_list in group_results:
#     uid_ngrams.extend([{'uid':uid, 'ngram':ngram} for uid,ngram in ngram_list])
# pd.DataFrame(uid_ngrams)    

# %%

# %%
# %%time
# THIS CREATES AN INDEX FROM FIRST WORD NUMBER TO RELEVANT ROWS IN THE MERGED INDEX.
# SO, IF WE KNOW THE FIRST WORD OF THE NGRAM IS 16, val_to_index[16] will give a range of rows.
# AS MERGED_NGRAMS IS SORTED THIS IS A START-END RELATIONSHIP.

@jit(nopython=True)
def ngram_indexes_from_first_word(merged_ngram, total_words):
    val_to_index = np.ones((total_words,2),dtype=np.uint32) * -1
    for i, val in enumerate(merged_ngram[:,0]):
        if val_to_index[val,0] == -1:
            val_to_index[val,:] = [i,i]
        else:
            val_to_index[val,1] = i
    return val_to_index


word_to_ngram_indexes = ngram_indexes_from_first_word(allowed_ngrams, len(index_to_word))

# %%
# %%time
import scipy.sparse as sparse
import numpy as np
# ngram_per_uid = sparse.lil_matrix((len(SPEECH_INDEX), merged_ngram.shape[0]),dtype=np.uint16)

@jit(nopython=True)
def count_valid_ngrams(data2d, merged_ngram, word_to_ngram_indexes, allowed_per_col):
    valid_ngrams = np.ones(data2d.shape[0],dtype=np.bool_)
    # First we do a quick filter for allowed words
    for col in range(data2d.shape[1]):
        valid_ngrams = valid_ngrams & allowed_per_col[:,col][data2d[:,col]]
    data2d = data2d[valid_ngrams,:]
    found_ngram = np.ones(data2d.shape[0], np.uint32) * -1
        
    for idx, row in enumerate(data2d):
        
        start_index, end_index = word_to_ngram_indexes[row[0]]
        
        # print(start_index, ' ', end_index)
        if start_index == -1:
            continue
        # Merged is always sorted. So, limiting range this way should work.
        for col in range(1,merged_ngram.shape[1]): # Start at col2, we know col1 matches
            row_val = row[col]
            col_vals = merged_ngram[start_index:end_index,col] # get current matches
            start = -1
            end = -1
            
            # First find start, if any
            for i, val in enumerate(col_vals):
                if val == row_val:
                    start = i + start_index
                    break
            # Not found at all, exit and continue with next ngram                    
            if start == -1: 
                start_index = -1
                break
            # Find end
            for i,val in enumerate(merged_ngram[start:end_index,col]):
                
                if val != row_val:
                    # Found the end
                    end = start + i # Verify this works
                    break
            start_index = start
            if end != -1:
                end_index = end # Only change if end was found
            
        if start_index != -1:
            found_ngram[idx] = start_index
    return found_ngram[found_ngram > -1]

def __count_valid_ngrams(chunk):
    vectors = []
    uids = vectorized_tex_df.u_id.values[chunk]
    chunk_text = vectorized_tex_df.vectorized_text.values[chunk]
    ngram_list = []
    for uid, data in zip(uids,chunk_text):
        if len(data) < ngram_length:
            continue
        data2d = np.lib.stride_tricks.sliding_window_view(data, window_shape=ngram_length)
        valid_ngrams = count_valid_ngrams(data2d, allowed_ngrams, word_to_ngram_indexes, allowed_per_col)
        ngram_list.append({'uid':uid,'ngrams':valid_ngrams})
    return ngram_list

def threaded_valid_ngram_counting(threads):
    chunks = np.array_split(range(len(vectorized_tex_df)),threads)
    ngrams_per_uid = []
    with Pool(threads) as pool:
        ngrams_per_uid = pool.map(__count_valid_ngrams, chunks)
    return_list = []
    for ng in ngrams_per_uid:
        return_list.extend(ng)
    return return_list

ngrams_per_uid = threaded_valid_ngram_counting(24)

# %%
# %%time
from collections import defaultdict

from scipy.sparse import lil_matrix, coo_matrix, dok_matrix
uid_ngram_sparse = dok_matrix((len(ngrams_per_uid), len(allowed_ngrams)),dtype=np.uint16).tolil()
for idx, uid_ngrams in enumerate(ngrams_per_uid):
    ngrams= uid_ngrams['ngrams']
    if len(ngrams) == 0:
        continue
    d = defaultdict(int)
    for ngram in ngrams:
        d[ngram] += 1
    uid_ngram_sparse.data[idx] = list(d.values())
    uid_ngram_sparse.rows[idx] = list(d.keys())

# %% [markdown]
# # WE HAVE THE NGRAMS IN SPARSE_MATRIX, u_id BY ngram
# ## Ngrams in allowed_ngrams
# ## Converting to text in index_to_word

# %%
uid_ngram_sparse
allowed_ngrams
    

# %%
def ngram_to_word(ngram):
    return ' '.join([index_to_word[x] for x in ngram])

def get_documents_using_ngram_index(index):
    uid_indexes =  uid_ngram_sparse.T.rows[index]
    if len(uid_indexes) == 0:
        return None
    uids = [ngrams_per_uid[x]['uid'] for x in uid_indexes]
    return SPEECH_INDEX[SPEECH_INDEX.u_id.isin(uids)]
              
print(ngram_to_word(allowed_ngrams[1001,:]))
df = get_documents_using_ngram_index(1001)
df.iloc[1].text_merged

 # %%
 == 'i-3f3da085571f7489-0']

# %%

# %%
# # %%time
# def __create_hashes(chunk):
#     start, end = chunk
#     print(f'Starting hash ({start} - {end})')
#     chunk_hashes = set()
#     ngrams = merged_ngram[start:end+1,:]
#     for row in ngrams:
#         chunk_hashes.add(hash(tuple(row)))
#     print('Finished chunk')
#     return chunk_hashes

# def threaded_hashing(threads):
#     chunks = np.array_split(range(merged_ngram.shape[0]),threads)
#     # chunks = np.array_split(range(tst.shape[0]),threads)
#     chunks = [(x[0],x[-1]) for x in chunks]
#     print(chunks)
#     all_hashes = []
#     with Pool(threads) as pool:
#         for result in pool.imap_unordered(__create_hashes, chunks):
            
#             print('joining...')
#             all_hashes.append(result)
#             # all_hashes = all_hashes.union(partial_hashes)
#     return all_hashes
# tst = np.array(range(100))

# all_hashes = threaded_hashing(12)
    

# %%
# %%time
for group_name, ngram_list in group_results:
    print(group_name)
    group_ngrams, group_count = np.unique(np.concatenate([x[1] for x in ngram_list if len(x[1]) > 0],axis=0),axis=0,return_counts=True)
    sorted_group_count = np.argsort(group_count)[::-1][:10]
    for i in sorted_group_count:
        converted = ' '.join([index_to_word[x] for x in group_ngrams[i]])
        print(group_count[i], ': ', converted)
    


# %%

# %%
import sys
local_vars = locals()
all_vars = []
for var_name in local_vars:
    if isinstance(local_vars[var_name], np.ndarray):
        var_size = local_vars[var_name].nbytes
    else:
        var_size = sys.getsizeof(local_vars[var_name])
    if var_size < 100000:
        continue
        
    all_vars.append((var_name, var_size / 1024 / 1024))
sarr = sorted(all_vars, key=lambda x: x[1],reverse=True)    
for name, size in sarr:
    print(name, ' ', size)

# %%
ngrams

# %%
np.unique(merged_ngram[:,0]).shape

# %%
merged_ngram.nbytes / 1024 / 1024

# %%

# %%

# %%
# arr1 = np.array([[1, 2], [3, 4], [5, 6]])
# arr2 = np.array([[1, 2], [4, 5], [5, 6],[7, 8]])
# print(arr1.shape)
# arr1c = np.array([1,2,3])
# arr2c = np.array([1,2,3,4])
# r, c = merge_sorted_arrays(arr1,arr2,arr1c,arr2c)
# print(r)
# print(c)

# %%

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
group_int_texts[('M', 1960)].shape
