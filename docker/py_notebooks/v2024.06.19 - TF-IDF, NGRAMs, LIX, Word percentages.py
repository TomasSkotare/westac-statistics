# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys

# Note that we do not install westac_statistics here - we use the files directly to more easily be able to change them.

# !{sys.executable} -m pip install nltk pyarrow openpyxl plotly  kaleido pyriksprot

# %%
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %%
import sys
sys.path.append('../') # To allow for import of westac_statistics
from westac_statistics import corpus_fast_tokenizer as cft
import pandas as pd
from IPython.display import display, Markdown, Image

corpus_version = 'v2024.06.19'
SPEECH_INDEX = pd.read_feather(f'./output/{corpus_version}/speech_index_{corpus_version}.feather')
SPEECH_INDEX = cft.FastCorpusTokenizer.prepare_dataframe(SPEECH_INDEX)
output_path = f'./output/{corpus_version}/'
temporary_path = f'./output/{corpus_version}/temp/'


# %%
# Helper class in case it's needed
def get_class_variables_info(obj):
    data = []
    for attr_name, attr_value in obj.__dict__.items():
        length = len(attr_value) if isinstance(attr_value, (str, list, np.ndarray)) else None
        value = attr_value if (length is None and isinstance(attr_value, (int, float))) else None
        shape = attr_value.shape if isinstance(attr_value, np.ndarray) else None
        data.append([attr_name, type(attr_value), length, value, shape])
    df = pd.DataFrame(data, columns=['Variable Name', 'Variable Type', 'Variable Length', 'Value', 'Shape'])
    return df


# %%
MD = lambda x: display(Markdown(x))

# %%
MD('# WE REMOVE ALL SPEECHES FROM BEFORE 1920!')
SPEECH_INDEX = SPEECH_INDEX[SPEECH_INDEX.year >= 1920].reset_index()

# %%
# %%time
import importlib
importlib.reload(cft)  
from westac_statistics import corpus_fast_tokenizer as cft


USE_STOP_WORDS = False

if USE_STOP_WORDS:
    stop_word_file = '../reference_files/predetermined_stop_words.xlsx'
else:
    stop_word_file = None

my_cft = cft.FastCorpusTokenizer(SPEECH_INDEX,threads = 24, stop_word_file=stop_word_file, minimum_ngram_count=10, regex_pattern=R"(?u)\b\w+\b") # We allow words of length 1!


# %%
##Sanity check, should be 0!
import numpy as np
np.sum(np.asarray(np.sum(my_cft.NGRAM_SPARSE_COUNT,axis=0)).flatten() < my_cft.MINIMUM_NGRAM_COUNT)

# %%
# TF-IDF på endast 7-grams som börjar på ordet "att"
# Ordfrekvens: Räkna ut vilka ord är de mest vanliga, använd den listan för att sedan se hur många ord som behövs för att täcka in 10, 50, 80 och 90% av texten (tror jag ni menar...)
# Långa ord: För varje "grupp" (decade/party) kolla hur många procent är (5,10,15,20,25)+ bokstäver
# Lix: enligt formel, grupp som ovan

# %%
# %%time
from tqdm.auto import tqdm
import importlib
from westac_statistics import tf_idf_calculator as tfidf
importlib.reload(tfidf)  
from westac_statistics import tf_idf_calculator as tfidf


tfidf_calc = tfidf.TF_IDF_Calculator(my_cft)
# This is where we decide the groups to split, i.e. party/decade.
data = tfidf_calc.calculate_ngram_groups(['decade'])

# %%
# THIS PARTS REMOVE ALL WORDS NGRAMS BUT THE ONES CONTAINING 'att' AT THE START
# word_index = my_cft.CONVERT_DICT['att']
# start, end = my_cft.WORD_TO_NGRAM_INDEXES[word_index,:]

# # SKip all but those starting with 'att'
# data = tfidf_calc.calculate_ngram_groups(['party_abbrev','decade'], ngram_allowed_indexes=(start,end))

# %%
# %%time
df_dict = tfidf_calc.group_data_to_dataframe(data, sort_by='count_idf')

# %%
# df_dict

# %%
names_to_index, ngram_total_usage_per_group, ngram_used_in_group = tfidf_calc.count_group_usages(data)
ngram_total_usage_per_group.shape

# %%
# %%time
MD("# TTR Calculations")
ttr = my_cft.ttr_function()
my_cft.SPEECH_INDEX['ttr'] = ttr
ttr_df = my_cft.SPEECH_INDEX.groupby(['party_abbrev','decade']).agg({'ttr':['mean', 'min', 'max', 'std']})

# %%
ttr_df.reset_index().to_excel(f'{output_path}/ttr_party_decade.xlsx')

# %%
from pathlib import Path
import plotly.express as px
def create_ttr_ranges(my_cft, token_ranges = [(20,200), (201,500), (501, np.inf)]):
    speech = my_cft.SPEECH_INDEX
    for _min, _max in token_ranges:
        range_data = speech[(speech.n_tokens >= _min) & (speech.n_tokens <= _max)]

        excel_file = f'./output/{corpus_version}/TTR_Over_Time.xlsx'

        #Average TTR per Gender over Time
        data = range_data.groupby(['year','gender'])['ttr'].mean().to_frame().unstack(level=0).T
        data.index = [year for tokens, year in data.index]
        if not Path(excel_file).exists(): # We need to create the file first if it does not exist
            data.to_excel(excel_file,sheet_name=f'Per gender ({_min} to {_max})')
        else:
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            # with pd.ExcelWriter('TTR_Over_Time.xlsx', engine='openpyxl', mode='w') as writer:
                data.to_excel(writer,sheet_name=f'Per gender ({_min} to {_max})')
        plot = px.line(data,title=f'Average TTR per gender over Time ({_min} to {_max})', labels={'index':'Year','value':'TTR'},height=600)
        display(plot.update_layout(hovermode='x'))

        #Average TTR per Party over Time
        data = range_data.groupby(['year','party_abbrev'])['ttr'].mean().to_frame().unstack(level=0).T
        data.index = [year for tokens, year in data.index]
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            data.to_excel(writer,sheet_name=f'Per party ({_min} to {_max})')
        plot = px.line(data,title=f'Average TTR per party over Time ({_min} to {_max})', labels={'index':'Year','value':'TTR'},height=600)
        display(plot.update_layout(hovermode='x unified'))

        #Average TTRover Time
        data = range_data.groupby(['year'])['ttr'].mean()
        # data.index = [year for tokens, year in data.index]
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            data.to_excel(writer,sheet_name=f'Mean per year ({_min} to {_max})')
        plot = px.line(data,title=f'Average TTR over Time ({_min} to {_max})', labels={'index':'Year','value':'TTR'},height=600)
        plot.update_layout(hovermode='x')
MD("# Save mean TTR over time for the different token categories")
create_ttr_ranges(my_cft)

# %%
MD("# Spara undan högst och lägst TTR per decenium (50 st top/bottom)")
MD("## Hela dataframen (för match), en tab per decenium en fil per top/bottom")
MD("## EN GÅNG PER GRUPP i.e. 20-200, 201-500, 501-1000, 1001+")


# %%
def save_top_ttr_per_decade(my_cft, token_ranges = [(20,200), (201,500), (501, 1000), (1001, np.inf)], top_bottom_count = 50):
    speechdf = my_cft.SPEECH_INDEX
    def get_excel_writer_args(excel_file):
        if Path(excel_file).exists():
            extra_args = {'mode':'a', 'if_sheet_exists':'replace'}
        else:
            extra_args = {'mode':'w'}
        return extra_args

    for _min, _max in token_ranges:
        excel_name = f'./output/{corpus_version}/TTR_Top_{top_bottom_count}_({_min}-{_max}_tokens).xlsx'

        range_data = speechdf[(speechdf.n_tokens >= _min) & (speechdf.n_tokens <= _max)]
        # range_data['decade'] = (range_data.year / 10).astype(int) * 10
        for decade in sorted(range_data.decade.unique()):
            # Keep high values at the beginning!
            decade_data = range_data[range_data.decade == decade].sort_values(by='ttr',ascending=False)
            # Change writing mode to w if file does not exist!
            args = get_excel_writer_args(excel_name)
            with pd.ExcelWriter(excel_name, engine='openpyxl', **args) as writer:
            # with pd.ExcelWriter(f'TTR_Top_{top_bottom_count}_({_min}-{_max}_tokens).xlsx', engine='openpyxl', mode='w') as writer:
                decade_data.head(top_bottom_count).to_excel(writer,sheet_name=f'{decade}')
            with pd.ExcelWriter(excel_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:            
            # with pd.ExcelWriter(f'TTR_Bottom_{top_bottom_count}_({_min}-{_max}_tokens).xlsx', engine='openpyxl', mode='w') as writer:
                decade_data.tail(top_bottom_count).to_excel(writer,sheet_name=f'{decade}')        
save_top_ttr_per_decade(my_cft)

# %%
# vi behöver en uppdaterad TTR-graf från dig baserat på version 0.10.0. Denna scatter plot-graf ska innehålla:
#genomsnittlig TTR för fyra tallängdkategorier: 20–200 ord, 201–500 ord, 501–1000 ord och 1000+ ord
#genomsnittlig TTR ska vara uträknad per decennium från 1920-talet till och med 2010-talet
#bara för partierna C, KD, L, M, MP, NYD, S, SD och V. I tidigare versioner har vi även tagit bort partier benämnda som FRIS, X och ? men dessa ska nu tas bort
#färgkoda partierna efter deras officiella partifärger (eller iaf så likt det går)
#och ta gärna bort 2020-talet, det är bara fram till 2019 vi är intresserade av
#använd grå färg för Nyd
import plotly.io as pio
import os
from westac_statistics import case_one_gui
importlib.reload(case_one_gui)  
from westac_statistics import case_one_gui

def resize_and_save_plotly_figure(fig, filename, output_types=['png', 'svg', 'html', 'pdf'], output_path='', width=None, height=None):
    # Resize the figure
    fig.layout.autosize = False
    if width is not None:
        fig.layout.width = width
    if height is not None:
        fig.layout.height = height

    # Save the figure in the specified output types
    for output_type in output_types:
        if output_type in ['png', 'svg','pdf']:
            pio.write_image(fig, os.path.join(output_path, f'{filename}.{output_type}'))
        elif output_type == 'html':
            fig.write_html(os.path.join(output_path, f'{filename}.html'))

def ttr_per_decade_per_party_per_token_range(s_df, token_ranges = [(1,20,200), (2,201,500), (3,501, 1000), (4,1001, np.inf)], 
                                             parties_to_include=None):
    color_map = None
    temp_df = s_df.copy()
    temp_df = temp_df.rename(columns={'party_abbrev': 'Party', 'ttr':'TTR'})

    # Strip parties that are not included
    if parties_to_include is not None:
        temp_df = temp_df[temp_df.Party.isin(parties_to_include.party_abbreviation)]
        color_map = dict(zip(parties_to_include.party_abbreviation, parties_to_include.color))

    temp_df['token_range'] = temp_df.n_tokens.apply(lambda x: next(iter([f'{n}: {a}-{b}' for n,a,b in token_ranges if ((x >= a) & (x <= b))]),'None'))
    temp_df = temp_df[temp_df.token_range != 'None']
    temp_df['decade'] = [f'{x}s' for x in temp_df.year.apply(lambda x: int(x / 10) * 10)]
    temp_df = temp_df.groupby(['decade','Party','token_range']).agg({'TTR':'mean'})
    fig = px.strip(temp_df.reset_index(), x='decade',y='TTR',color='Party',facet_col='token_range',height=800, 
                   title='TTR per decade, split into party and different number of tokens in speech',
                   color_discrete_map=color_map, 
                   labels={
                     "ttr":"TTR",
                   },)
    # A variant for updating the subplot titles
    # for a in range(len(token_ranges)):
    #     i, start, stop = token_ranges[a]
    #     fig.add_annotation(
    #         dict(
    #             text=f'Speech length = {start}-{stop} words',
    #             xref=f'x{a+1}',
    #             yref=f'y{a+1}',
    #             x=0.5,
    #             y=1.02,
    #             showarrow=False,
    #             font=dict(size=12),
    #             xanchor='center',
    #             yanchor='bottom'
    #         )
    #     )
    for a in range(len(token_ranges)):
        i, start, stop = token_ranges[a]
        fig.layout.annotations[a]['text'] = f'Speech length = {start}-{stop} words'
    
    fig.update_xaxes(tickangle=45, nticks=len(temp_df))
    fig.update_layout(legend_title_text='Party')
    
    return fig, temp_df


color_df = pd.DataFrame(tuple(case_one_gui.CaseOneGUI.PARTY_COLORS_HEX.items()),columns=['party_abbreviation', 'color'])
color_df.party_abbreviation = color_df.party_abbreviation.str.title()
specified_set = {'C', 'Kd', 'L', 'M', 'Mp', 'Nyd', 'S', 'Sd', 'V'}
color_df = color_df[color_df.party_abbreviation.isin(specified_set)]

fig, temp_df = ttr_per_decade_per_party_per_token_range(my_cft.SPEECH_INDEX[my_cft.SPEECH_INDEX.decade < 2020], 
                                                        parties_to_include=color_df)
display(fig)
# Now use the resize_and_save_plotly_figure function to save the figure
resize_and_save_plotly_figure(fig, '2024_01_15-ttr_ranges_over_decades', output_types=['png', 'svg', 'html','pdf'], 
                              output_path=output_path, width=1000, height=600)

# color_df

# %%
pd.DataFrame(tuple(case_one_gui.CaseOneGUI.PARTY_COLORS_HEX.items()),columns=['party_abbreviation', 'color'])

# %%
my_cft.SPEECH_INDEX.party_abbrev.unique()

# %%
# %%time
# # Ordfrekvens: Räkna ut vilka ord är de mest vanliga, använd den listan för att sedan se hur många ord som behövs för att täcka in 10, 50, 80 och 90% av texten (tror jag ni menar...)
most_used = my_cft.get_most_used_words_per_group(['party_abbrev','decade'])
group_cumulative_sums = my_cft.word_use_cumulative_count(most_used)
my_cft.save_cumulative_sums_to_excel(group_cumulative_sums, output_path)


# %% [markdown]
# # Strip plot of TTR per decade per party, split into token count categories

# %%
def ttr_per_decade_per_party_per_token_range(s_df, token_ranges = [(1,20,50), (2,51,100), (3,101,150), (4,151,200), (5,201,500), (6,501, 1000), (7,1001, np.inf)]):
    temp_df = s_df.copy()
    temp_df['token_range'] = temp_df.n_tokens.apply(lambda x: next(iter([f'{n}: {a}-{b}' for n,a,b in token_ranges if ((x >= a) & (x <= b))]),'None'))
    temp_df = temp_df[temp_df.token_range != 'None']
    temp_df['decade'] = temp_df.year.apply(lambda x: int(x / 10) * 10)
    temp_df = temp_df.groupby(['decade','party_abbrev','token_range']).agg({'ttr':'mean'})
    fig = px.strip(temp_df.reset_index(), x='decade',y='ttr',color='party_abbrev',facet_col='token_range',height=800, title='TTR per decade, split into party and different number of tokens in speech')
    
    temp_df.to_excel(f'{output_path}/ttr_per_decade_per_party_per_token_range.xlsx')
    # del temp_df
    return fig, temp_df

# NOTE THAT WE ONLY TAKE YEARS AFTER 1910 HERE AS IT IS NOT RELEVANT BEFORE
plt, temp_df = ttr_per_decade_per_party_per_token_range(my_cft.SPEECH_INDEX[my_cft.SPEECH_INDEX.decade > 1910])
plt


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
MD('"# LIX over time per party; displayed mostly as a sanity check.')
px.line(df, x='Decade',y='LIX (mean)',color='Party')

# %%
MD("# Most common sentances for high (>0.95) TTR numbers!")
for i, (text, count) in enumerate(SPEECH_INDEX[(SPEECH_INDEX.ttr > 0.95) & (SPEECH_INDEX.n_tokens >= 20)].text_merged.value_counts().items()):
    print(f'{count}:\t{text}')
    if i > 20:
        break

# %%
text, count = next(SPEECH_INDEX[(SPEECH_INDEX.ttr > 0.95) & (SPEECH_INDEX.n_tokens >= 20)].text_merged.value_counts().items())
MD('## Top years for long speeches with high TTR, while still being at least 20 tokens long')
SPEECH_INDEX[SPEECH_INDEX.text_merged == text].year.value_counts()


# %% [markdown]
# # Violin plots of difference in TTR between genders over time

# %%
import plotly.graph_objects as go

df = SPEECH_INDEX[SPEECH_INDEX.n_tokens < 10000].copy()
df['decade'] = df.year.apply(lambda x: int(x/10) * 10)

token_ranges = [(20,200), (201,500), (501, 1000), (1001, np.inf)]

df['token_range'] = df.n_tokens.apply(lambda x: next(iter([f'{a}-{b}' for a,b in token_ranges if ((x >= a) & (x <= b))]),'None'))

def sort_order(string):
    try:
        ret_val = int(string.split('-')[0])
        # print(ret_val)
        return ret_val
    except:
        return np.inf
    

for r in sorted(df.token_range.unique(), key=lambda x: sort_order(x)):
    fig = go.Figure()
    _df = df[df.token_range == r]
    genders = {'man':  {'color':'blue',  'side':'negative'},
               'woman':{'color':'orange','side':'positive'}}
    for gender, args in genders.items():
        fig.add_trace(        
            go.Violin(x=_df['decade'][_df.gender == gender], 
                      y=_df['ttr'][_df.gender == gender], 
                      legendgroup=gender, scalegroup=gender, name=gender,
                      side=args['side'], line_color=args['color']))
    fig.update_traces(meanline_visible=True)
    fig.update_layout(title=f'Token range: {r}', xaxis=dict(tickvals=_df['decade'].unique()))
    # fig.update_layout(violingap=0.01, violinmode='overlay')

    display(fig)
df = None


# %%
# Plot of number of speeches in different ranges
def number_of_speeches_per_token_range(s_df, token_ranges = [(1,20,200), (2,201,500), (3,501,1000), (4,1001, np.inf)]):
    temp_df = s_df.copy()
    temp_df['token_range'] = temp_df.n_tokens.apply(lambda x: next(iter([f'{n}: {a}-{b}' for n,a,b in token_ranges if ((x >= a) & (x <= b))]),'None'))
    temp_df = temp_df[temp_df.token_range != 'None']
    temp_df['decade'] = temp_df.year.apply(lambda x: int(x / 10) * 10)
    # temp_df = temp_df.groupby(['decade','party_abbrev','token_range']).agg({'ttr':'mean'})
    return temp_df
display(Markdown("# Plots regarding TTR over time"))    
df = number_of_speeches_per_token_range(SPEECH_INDEX)
display(px.line(df.groupby(['year','token_range']).ttr.mean().unstack(level=0).T,title='Average TTR per token range over years'))
display(px.area(df.groupby(['year','token_range']).ttr.count().unstack(level=0).T,title='Number of speeches in a token range by year by fraction',height=600, groupnorm='fraction'))
display(px.line(df.groupby(['year','token_range']).ttr.count().unstack(level=0).T,title='Number of speeches in a token range by year',height=600))

display(Markdown("# List number of speeches per token range per decade"))
df[df.year >= 1920].groupby(['year','token_range']).ttr.count().unstack(level=0).T.to_excel(f'{output_path}/number_of_speeches_per_token_count_per_year.xlsx')

# %%
fig = px.scatter(df,x='ttr',y='n_tokens',marginal_x="histogram", marginal_y="histogram",size_max=1)
# fig.update_traces(marker={'size': 1})
image_location = f'{output_path}/test_image.png' 
fig.write_image(image_location)
display(Image(f'{output_path}/test_image.png'))

# %% [markdown]
#
# fig = px.density_contour(df,x='ttr',y='n_tokens',height=600, width=600)
# fig

# %%
import matplotlib.pyplot as plt
import numpy as np

def get_ttr_by_decade_distribution(df):
    # Define the limits for x and y
    xlim = (0, 100)
    ylim = (0, 12000)
    size_modifier = 4

    # Create a 4x4 grid of subplots
    shape = (5,4)
    fig, axs = plt.subplots(shape[0], shape[1], figsize=(3*shape[1], 3*1.15*shape[0]), sharex=True, sharey=True)

    # # Set the common labels
    fig.text(0.5, 0.04, 'TTR', ha='center', va='center')
    fig.text(0.06, 0.5, 'Number of tokens', ha='center', va='center', rotation='vertical')


    # plt.show(



    for _num, decade in enumerate(sorted(df.decade.unique())):
        # print(decade)
        # fig = matplotlib.pyplot.figure();
        j = _num % shape[1]
        i = int(_num / shape[1])
        # print(i, ' ', j)
        mappable = axs[i, j].hexbin(df[df.decade == decade].ttr, df[df.decade == decade].n_tokens,
                                 bins='log', gridsize=40,
                                extent=(0,100,0,12000));
        axs[i, j].set_xlim(xlim)
        axs[i, j].set_ylim(ylim)
        axs[i, j].set_title(f'{decade}')
        # axs[i, j].margins(x=0.15, y=0.15)
        # axs[i, j].set_aspect('box')
        axs[i, j].set_box_aspect(1)


    # Adjust the subplots to make room for the colorbar
    fig.subplots_adjust(right=0.8)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.85, 0.35, 0.02, 0.3])
    cbar_ax.tick_params(labelsize=0, length=0)
    cbar = fig.colorbar(mappable, cax=cbar_ax, shrink=0.2)    
    cbar.minorticks_off()   
    cbar.set_label('Relative density (log)', rotation=270, labelpad=20)
    return fig
MD("""# Plot of TTR vs number of tokens, by decade

This is intended to show how the distribution of TTR vs length changes over time.

We can see that there is a wider spread in how long and the TTR of speeches earlier, and it becomes more normlized as time goes on.

Note that the colors cannot be compared between decades, as each decade has density normalization completely separate from the others.""")

fig = get_ttr_by_decade_distribution(df)
fig.savefig(f'{output_path}/ttr_by_number_of_tokens_per_decade.pdf')


# %%
from tqdm.auto import tqdm
MD('# Till mig och Johan, skulle du kunna ta fram en beräkning på\n ## 1) hur många totalt unika ord det är för decennium och för parti, \n## 2) en ttr för varje decennium (och inte fördelat på parti)?')


# MERGE THIS WITH SPEECH
vectorized = my_cft.VECTORIZED_TEX_DF
number_of_words = my_cft.word_count

df = SPEECH_INDEX
df['decade'] = df.year.apply(lambda x: int(x/10) * 10)
data_list = []
for (party, decade), group in tqdm(SPEECH_INDEX.groupby(['party_abbrev','decade'])):
    word_count = np.zeros(number_of_words,dtype=np.uint32)
    vectorized_words_spoken = np.concatenate(vectorized.loc[group.index].vectorized_text.values)
    np.add.at(word_count, vectorized_words_spoken, 1)
    data_list.append((party, decade, np.sum(word_count > 1)))


from contextlib import contextmanager

@contextmanager
def custom_format(format_string):
    original_format = pd.options.display.float_format
    pd.options.display.float_format = format_string
    try:
        yield
    finally:
        pd.options.display.float_format = original_format

party_df = pd.DataFrame(data_list, columns = ['Party','Decade','Unique words'])
party_df = party_df.pivot(index='Party', columns='Decade', values='Unique words')
MD('# Number of unique words spoken per party per decade')
with custom_format('{:.0f}'.format):
    display(party_df)
    party_df.to_excel(f'{output_path}/unique_words_spoken_per_party_per_decade.xlsx')


with custom_format('{:.0f}'.format):
    MD('Mean TTR per decade and party')
    display(df[['party_abbrev','decade','ttr']].groupby(['party_abbrev', 'decade']).agg('mean').reset_index().pivot(index='party_abbrev', columns='decade', values='ttr'))

MD('Mean TTR per decade')
df.groupby('decade').ttr.mean().to_frame()


# %%
#En annan grej som du skulle kunna hjälpa oss med: ta fram topp 1000 listor med de vanligaste längst orden (definierat som ord som innehåller minst 16 tecken) för varje decennium
#Behöver inte vara fördelat på parti

def get_top_words(output_path, top_number=1000, min_word_length=16):
    for name, df in SPEECH_INDEX.groupby('decade'):
        word_count = np.zeros(my_cft.word_count, dtype=np.uint32)
        words = my_cft.VECTORIZED_TEX_DF.iloc[df.index]
        for text in words.vectorized_text:
            np.add.at(word_count, text, 1)
        word_count[my_cft.WORD_LENGTH < min_word_length] = 0
        top_indices = np.argsort(word_count)[::-1][:top_number]
        top_df = pd.DataFrame({'Word': [my_cft.INDEX_TO_WORD[x] for x in top_indices], 'Count': word_count[top_indices]})
        top_df.to_excel(f'{output_path}/{name}s_top_{top_number}_words.xlsx')
    
get_top_words(output_path, top_number=1000, min_word_length=16)

# %%

# %%
my_cft.
