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

import __paths__  # pylint: disable=unused-import
from dataclasses import dataclass
from westac_analysis.case_one_gui import CaseOneGUI
from westac_analysis.common import load_speech_index

# %load_ext autoreload
# %autoreload 2

import ipywidgets
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
from IPython.display import display

import plotly.express as px


# %%
import sqlite3

con = sqlite3.connect('./tagged_frames_v0.4.1_speeches.feather/riksprot_metadata.main.db')
cursor = con.cursor()

# %%
sql_query = """SELECT name FROM sqlite_master  
  WHERE type='table';"""
table_names = [x[0] for x in cursor.execute(sql_query).fetchall()]

# %%

# %%
all_tables = {}
for table in table_names:
    print(table)
    all_tables[table] = pd.read_sql(f"select * from {table}", con=con)

# %%
for table_name, df in all_tables.items():
    print(table_name)
    display(df.head(3))

# %%
SPEECH_INDEX[SPEECH_INDEX.who=='Q4820820']

# %%
df = all_tables['_party_affiliation']
df[df.person_id =='Q4820820' ]

# %%
df = all_tables['_person']
# df
df[df.person_id =='Q4820820' ]

# %%

# %%
import plotly.express as px
df = all_tables['_person']

# px.bar(df.groupby('gender').count(), barmode='group')
fig = px.bar(df.groupby('gender').count()[['person_id', 'dead', 'riksdagen_id']],
       height=600,
       barmode='group', 
       labels={'gender':'Kön','value':'Antal','variable':'Datapunkt'},
            title='Översikt')
# print(fig)
translate = {'person_id':'Totalt antal', 'dead':'Avlidna', 'riksdagen_id':'Antal i riksdagen'}
fig.for_each_trace(lambda t: t.update(name = translate[t.name],
                                      legendgroup = translate[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, translate[t.name])
                                     )
                  )
fig

# %%
all_tables['party']

# %%
df = all_tables['_minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.join(all_tables['_person'], lsuffix='_left', rsuffix='_right')
df = df.drop(columns=[x for x in df.columns if x.endswith('_right')])
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['person_id_left'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['Total'] = df2.agg(sum,axis=1)
df2 = df2.sort_values(by='Total',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by total number and grouped by gender')

# %%
df = all_tables['_minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.join(all_tables['_person'], lsuffix='_left', rsuffix='_right')
df = df.drop(columns=[x for x in df.columns if x.endswith('_right')])
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['person_id_left'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['agg'] = df2.agg(lambda x: np.abs(np.diff(x)),axis=1)
# display(df2)
df2 = df2.sort_values(by='agg',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by largest difference first and grouped by gender')

# %%
df = all_tables['_minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.merge(all_tables['_person'], on='person_id')
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['person_id'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['agg'] = df2.agg(lambda x: np.abs(np.diff(x)),axis=1).fillna(-10000)
# display(df2)
df2 = df2.sort_values(by='agg',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by largest difference first and grouped by gender')

# %%

# %%
px.bar(all_tables['unknown_utterance_gender'].set_index('gender_id').join(all_tables['gender'].set_index('gender_id'),how='inner')['gender'].value_counts(), title='Gender of unknown speakers', labels={'value':'Count', 'index':'Gender'})

# %%

# %%
all_tables['utterances'].speaker_hash.nunique()

# %%
all_tables['utterances'].groupby('document_id').speaker_hash.value_counts().max()

# %%
px.histogram(sorted(all_tables['utterances'].groupby('document_id').speaker_hash.nunique(),reverse=True))

# %%
pd.DataFrame(all_tables['utterances'].groupby('document_id').speaker_hash)

# %%
for x in all_tables.keys():
    print(x)

# %%
df = all_tables['_unknowns']
df[df.hash=='3d928ecb']
# df[df.u_id == 'i-2bc16c2052160500-0']

# %%
df = all_tables['_person']
df


# %%
def join_on(df1, df2, column, delete_join = False):
    len_before = len(df1)
    df = df1.set_index(column).join(df2.set_index(column), how='left').reset_index()
    if len(df) != len_before:
        print(f'Number of rows changed, before: {len_before:10}, after: {len(df):10}, difference: {(len(df) - len_before):10}')
    if delete_join:
        return df.drop(columns=column)
    else:
        return df


# %%
df = all_tables['chamber']
display(df)

def apply_dict(val, dict, default_value=np.nan):
    try:
        return dict[val]
    except KeyError as e:
        return default_value
            

# SPEECH_INDEX['chamber_id'].apply(lambda x: apply_dict(x, convert_dict, default_value='nan'))

# %%
SPEECH_INDEX = pd.read_feather('./tagged_frames_v0.4.1_speeches.feather/document_index.feather')
SPEECH_INDEX = join_on(SPEECH_INDEX, all_tables['party'], 'party_id', delete_join=True)
SPEECH_INDEX = join_on(SPEECH_INDEX, all_tables['office_type'], 'office_type_id', delete_join=True)
SPEECH_INDEX = join_on(SPEECH_INDEX, all_tables['sub_office_type'], 'sub_office_type_id', delete_join=True)
SPEECH_INDEX = join_on(SPEECH_INDEX, all_tables['gender'], 'gender_id', delete_join=True)
# SPEECH_INDEX = join_on(SPEECH_INDEX, all_tables['chamber'], 'chamber_id', delete_join=False)
convert_dict = all_tables['chamber'].set_index('chamber_id').to_dict()['chamber']      
SPEECH_INDEX['chamber'] = SPEECH_INDEX.chamber_id.apply(convert_dict.get)
# SPEECH_INDEX['chamber_id'].apply()
# DOCUMENT_INDEX.rename(columns='who', )
SPEECH_INDEX['file_name'] = SPEECH_INDEX.document_name.str.split('_').apply(lambda x: x[0])
SPEECH_INDEX

# %% [markdown]
# # Protocols with the largest number of tokens by unknowns

# %%
import plotly.graph_objects as go

df = SPEECH_INDEX[SPEECH_INDEX.who == 'unknown'].groupby(['file_name','gender']).agg({'n_tokens':sum}) #Change sum/len here!
# display(df)
df = df.join(df.groupby(['file_name']).agg({'n_tokens':sum}).rename(columns = {'n_tokens':'total_tokens'}),how='left').sort_values(by='total_tokens',ascending=False)
df = df.head(100).drop(columns='total_tokens').unstack(1)
df.columns = [x[1] for x in df.columns]
display(px.bar(df, barmode='stack',height=600))

import glob
xml_files = list(glob.iglob('./corpus_compressed/protocols/**/*.xml', recursive=True))

def find_file(file):
    return next(iter([x for x in xml_files if x.__contains__(file)]),None)

df['file_path'] = list(map(find_file, iter(df.index.to_list())))
df

# %%
# ATTEMPT TO CHECK WHERE PARTY IS UNKNOWN
from ipywidgets import Output

df = SPEECH_INDEX[(SPEECH_INDEX.party_abbrev == '?')&(SPEECH_INDEX.who != 'unknown')].groupby(['file_name','gender']).agg({'n_tokens':len}) #Change sum/len here!
# display(df)
df = df.join(df.groupby(['file_name']).agg({'n_tokens':sum}).rename(columns = {'n_tokens':'total_tokens'}),how='left').sort_values(by='total_tokens',ascending=False)
df = df.head(100).drop(columns='total_tokens').unstack(1)
df.columns = [x[1] for x in df.columns]
fig = px.bar(df, barmode='stack',height=600,labels={'value':'Number of speeches', 'file_name':'Protocol name'},title='Protocols with  the most unknown party affiliations from known speakers')
fig.update_xaxes(type='category')
fig_widget = go.FigureWidget(fig.data, fig.layout)
display(fig)

import glob
xml_files = list(glob.iglob('./corpus_compressed/protocols/**/*.xml', recursive=True))

def find_file(file):
    return next(iter([x for x in xml_files if x.__contains__(file)]),None)

df['file_path'] = list(map(find_file, iter(df.index.to_list())))
df

temp_output = Output()
display(temp_output)

def do_things(x):
    with temp_output:
        print('test!')

fig_widget.data[0].on_click(do_things)


# %%
df = SPEECH_INDEX[(SPEECH_INDEX.year == 1992) & (SPEECH_INDEX.who == 'unknown')]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    display(pd.DataFrame(df.groupby(['file_name','party']).agg(
        **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum')})))

# %%
for file in df.file_path:
    print(file)

# %%
df.index.to_list()

# %%
all_tables['_member_of_parliament']


# %%
all_tables['protocols']

# %%

df = pd.read_feather('./tagged_frames_v0.4.1_speeches.feather/document_index.feather')
df['file_name'] = df.document_name.str.split('_').apply(lambda x: x[0])

# %%
df[df.gender_id == 0].groupby(['file_name']).size().sort_values(ascending=False).head(20)

# %%
all_tables['persons_of_interest']

# %% [markdown]
# # Number of speeches matching chambers:

# %%
temptemp = []
for n, count in join_on(SPEECH_INDEX, all_tables['chamber'], 'chamber_id', delete_join=True).chamber.value_counts().to_dict().items():
#         print(f'{n:20}: {count}')
        temptemp.append({'Chamber':n, 'Num':count})
temptemp.append({'Chamber':'NaN', 'Num':np.sum(np.isnan(SPEECH_INDEX.chamber_id))})
df = pd.DataFrame(temptemp)
px.bar(df.set_index('Chamber'),title='Number of speeches matching chambers')


# %% [markdown]
# # Total tokens per party:
#  - Plot is sorted in ascending order from left (few) to right (many)

# %%
px.bar(SPEECH_INDEX.groupby('party_abbrev')['n_tokens'].sum().sort_values(),title='Total tokens uttered per party',height=600)

# %%
import importlib
import westac_analysis.case_one_gui as cog
importlib.reload(cog)
PARTYS = SPEECH_INDEX.party_abbrev.unique().tolist()

guip = cog.CaseOneGUI(SPEECH_INDEX, filter_key='party_abbrev', filter_sub_key='gender')\
    .setup(filter_values=[''] + PARTYS)
display(guip.layout())
guip.update()

# %%
import importlib
import westac_analysis.case_one_gui as cog
importlib.reload(cog)

guip = cog.CaseOneGUI(SPEECH_INDEX, filter_key='party_abbrev', filter_sub_key='gender')\
    .setup(filter_values=[''] + PARTYS)
display(guip.layout())
guip.update()

# %%
# def count_empty(x):
#     return (x.isna() | x.isnull() | x.eq('') | x.eq('unknown')).sum()

# l = list()

# for col in MEMBERS.columns:
#     empty_percentage = round(count_empty(MEMBERS[col]) / len(MEMBERS['born'])*100,3)
#     if empty_percentage > 0:
#          l.append((col,empty_percentage))
# px.bar(pd.DataFrame(l,columns=['Value','Missing (%)']).set_index('Value').sort_values(by='Missing (%)'),
#        title='Amount of missing per column in data',labels={'Value':'Data column', 'value':'Missing (%)'},
#       height=600)

# %%

data = SPEECH_INDEX[SPEECH_INDEX.who == 'unknown']
display(data.head(5))
data = data.groupby(['year','gender']).n_tokens.sum()
px.line(data.unstack(level=0).T,height=600)


# %%
# TEST AV V
px.line(SPEECH_INDEX[SPEECH_INDEX.party_abbrev == 'V'].groupby('year').who.nunique())

# %% [markdown]
# # Average number of tokens spoken per year and gender

# %%
data = SPEECH_INDEX.groupby(['year','gender'])['n_tokens'].mean().to_frame().unstack(level=0).T
data.index = [year for tokens, year in data.index]

plot = px.line(data,labels={'index':'Year','value':'Average speeches per year'},
               title='Average number of tokens spoken per gender',
               height=600)
plot.update_layout(hovermode='x')

# %% [markdown]
# # Number of tokens in speeches by each member, by gender and year
# - NOTE: Completely silent members are not accounted for

# %%
import numpy as np
data = SPEECH_INDEX.groupby(['year','gender']).agg(
    **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum')})
data['Tokens per member and year'] = np.round(data['Number of tokens (total)'] / data['Number of speakers'],2)
# display(data.head(7))
labels = {'value':'Value','year':'Year'}
l1 = px.line(data['Number of speakers'].unstack(level=0).T,height=600, 
             title='Unique speakers per gender per year', labels=labels)
display(l1.update_layout(hovermode='x'))
l2 = px.line(data['Tokens per member and year'].unstack(level=0).T,
       title='Average number of tokens per speaker and gender per year',
       labels=labels, height=600)
display(l2.update_layout(hovermode='x'))
l3 = px.line(data['Number of speeches (total)'].unstack(level=0).T,
       title='Number of speeches per gender per year',
       labels=labels, height=600)
display(l3.update_layout(hovermode='x'))

data

# %%

# %%
# Additional info on unknown data:
SPEECH_INDEX[(SPEECH_INDEX.year == 2006)].groupby('gender').agg({'year':np.mean, 'n_tokens':np.sum, 'who':lambda x: len(np.unique(x))})

# %%
from ipywidgets import Tab
from IPython.display import display, HTML, DisplayHandle


from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea

from plotly.subplots import make_subplots
def create_per_party_plots(base_df: pd.DataFrame, column_to_show='Tokens per member and year', title=None):
    data = base_df.groupby(['year','gender','party_abbrev']).agg(
        **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum')})
    data['Tokens per member and year'] = np.round(data['Number of tokens (total)'] / data['Number of speakers'],2)

    party_list = data.index.get_level_values('party_abbrev').unique().to_list()
    fig = make_subplots(rows=len(party_list), 
                        cols=1,
                        shared_xaxes=True)
    fig.update_layout(hovermode='x')
    #                     subplot_titles=party_list)
    for i, party in enumerate(party_list):
        d = data[data.index.get_level_values('party_abbrev') == party][column_to_show].unstack(level=0).T
        d.columns = [g for g, p in d.columns]
        px_fig = px.line(d)
        for trace in px_fig.data:
            fig.append_trace(trace, row=i+1, col=1)

    fig.update_layout(height=1200, title_text=title)

    # This is a workaround to keep colors the same and not duplicate legend entries!
    possible_colors = px.colors.qualitative.Plotly
    genders = data.index.get_level_values('gender').unique()
    for i, g in enumerate(genders):
        for gender_plot in [x for x in fig.data if x.name == g]:
            gender_plot.line['color'] = possible_colors[i]

    for i, party in enumerate(party_list):
        fig.layout[f'yaxis{i+1}'].title=party

    remaining_genders = set(genders)
    for d in fig.data:
        if d.name in remaining_genders:
            remaining_genders.remove(d.name)
        else: 
            d.showlegend = False
    return fig

output_left = Output()
output_right = Output()
tabbed_interface = Tab(children=[output_left, output_right])
tabbed_interface.set_title(0,'Tokens per gender per party')
tabbed_interface.set_title(1,'Speakers per gender per party')
display(tabbed_interface)
with output_left:
    display(create_per_party_plots(SPEECH_INDEX, column_to_show='Tokens per member and year', title='Spoken tokens per gender per party'))
with output_right:
    display(create_per_party_plots(SPEECH_INDEX, column_to_show='Number of speakers', title='Number of unique speakers per party of different genders'))

# %%
year_group = SPEECH_INDEX.drop(columns='document_id').groupby('year')

plot = px.histogram(year_group.size(),height=600,
             labels={'index':'Year','value':'Average speeches per year'}, 
             title='Histogram of speeches per year ')
plot.update_layout(bargap=0.2)
plot
# ax.set_title('Histogram of number of speeches per year');

# %%
# join_on(SPEECH_INDEX.drop(columns=['year','document_name']), all_tables['protocols'], 'document_id')

display(all_tables['protocols'])

px.histogram(SPEECH_INDEX.n_tokens)

# %% [markdown]
# # Average age of speakers per party and gender

# %%
df = SPEECH_INDEX[SPEECH_INDEX.who != 'unknown'].set_index('who').join(all_tables['persons_of_interest'][['person_id','year_of_birth','year_of_death']].set_index('person_id'),how='inner')
df['who'] = df.index
df['approx_age'] = df.year - df.year_of_birth
# print('------ People speaking after death (first 5)')
# display(df[df.year > df.year_of_death].head(5))
print('------ People speaking after age 100 (first 5)')
display(df[df.approx_age > 100].head(5))


from ipywidgets import Tab
from IPython.display import display, HTML, DisplayHandle


from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea

from plotly.subplots import make_subplots
def create_per_party_ages(base_df: pd.DataFrame, column_to_show='Tokens per member and year', title=None):
    data = base_df.groupby(['year','gender','party_abbrev']).agg(
        **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum'), 'Average age':('approx_age','mean')})
    data['Tokens per member and year'] = np.round(data['Number of tokens (total)'] / data['Number of speakers'],2)
#     return data
    party_list = data.index.get_level_values('party_abbrev').unique().to_list()
    fig = make_subplots(rows=len(party_list), 
                        cols=1,
                        shared_xaxes=True)
    fig.update_layout(hovermode='x')
    #                     subplot_titles=party_list)
    for i, party in enumerate(party_list):
        d = data[data.index.get_level_values('party_abbrev') == party][column_to_show].unstack(level=0).T
        d.columns = [g for g, p in d.columns]
        px_fig = px.line(d)
        for trace in px_fig.data:
            fig.append_trace(trace, row=i+1, col=1)

    fig.update_layout(height=1200, title_text=title)

    # This is a workaround to keep colors the same and not duplicate legend entries!
    possible_colors = px.colors.qualitative.Plotly
    genders = data.index.get_level_values('gender').unique()
    for i, g in enumerate(genders):
        for gender_plot in [x for x in fig.data if x.name == g]:
            gender_plot.line['color'] = possible_colors[i]

    for i, party in enumerate(party_list):
        fig.layout[f'yaxis{i+1}'].title=party

    remaining_genders = set(genders)
    for d in fig.data:
        if d.name in remaining_genders:
            remaining_genders.remove(d.name)
        else: 
            d.showlegend = False
    return fig


display(create_per_party_ages(df, column_to_show='Average age', title='Average age of speaker per party'))
# output_left = Output()
# output_right = Output()
# tabbed_interface = Tab(children=[output_left, output_right])
# tabbed_interface.set_title(0,'Tokens per gender per party')
# tabbed_interface.set_title(1,'Speakers per gender per party')
# display(tabbed_interface)
# with output_left:
#     display(create_per_party_plots(SPEECH_INDEX, column_to_show='Tokens per member and year', title='Spoken tokens per gender per party'))
# with output_right:
#     display(create_per_party_plots(SPEECH_INDEX, column_to_show='Number of speakers', title='Number of unique speakers per party of different genders'))

# %%
list(all_tables.keys())

# %%
df = all_tables['_party_affiliation']
df[df.person_id=='Q98281504']

# %%
df = all_tables['_person']
df[df.person_id == 'Q5555684']

# %%
df = all_tables['_name']
df[df.person_id == 'Q5555684']

# %%
df = all_tables['person_multiple_party']
df[df.person_id=='Q98281504']

# %%
df = all_tables['person_yearly_party']
df[df.person_id=='Q98281504']

# %% [markdown]
# # People who are not unknown but have party unknown

# %%
df = SPEECH_INDEX[SPEECH_INDEX.who != 'unknown']
df[df.party == 'unknown'].who.unique()

# %%
SPEECH_INDEX[SPEECH_INDEX.who=='Q98281504']

# %%
df = join_on(all_tables['persons_of_interest'], all_tables['party'], 'party_id')
df = df[np.isnan(df.party_id)]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    display(df.sort_values(by='year_of_birth').head(10)) # REMOVE HEAD HERE IF YOU WANT ALL

# %%
### import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML, DisplayHandle


from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea

selected_documents = []

df = SPEECH_INDEX.set_index('document_name')['n_tokens']
# df = SPEECH_INDEX.groupby(['protocol_name'])['n_tokens'].sum()


fig = px.histogram(df, x="n_tokens", nbins=100)
fig.update_layout(bargap=0.2,
                  title='Histogram of number of tokens per protocol', 
                  xaxis_title='Number of tokens',
                  yaxis_title='Count',
                  height=600,
                  width=1000,
                 )
fig.update_traces(marker_color='rgb(158,202,225)', 
                  marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, 
                  opacity=0.6,
                  hovertemplate = 'Token range = %{x}<br>Count = %{y}<extra></extra>',
                 )

text_output = Textarea(
    placeholder = 'None selected',
    layout={'height': '500px'}
    )

df_output = Output()


def do_things(trace, points, state):
    selected_documents = [x for x,y in df.iloc[points.point_inds].items()]
    text_output.value = '\n'.join(selected_documents)
    df_output.clear_output()
    with df_output:
#         print(selected_documents)
#         display(SPEECH_INDEX[SPEECH_INDEX.protocol_name.isin(set(selected_documents))])
        display(SPEECH_INDEX[SPEECH_INDEX.document_name.isin(set(selected_documents))])
#         display(SPEECH_INDEX.head(10))
    

fig_widget = go.FigureWidget(fig.data, fig.layout)
# print(dir(fig))
fig_widget.data[0].on_selection(do_things)
fig_widget.data[0].on_click(do_things)

display(HBox([fig_widget, VBox([Label(f'Selected protocols (out of {len(df)})\n'), text_output])]))
display(df_output)

# %%
pd.read_feather('./tagged_frames_v0.4.1_speeches.feather/200607/prot-200607--114.feather')

# %%
import sys
sys.path.append('/usr/local/lib/python3.8/dist-packages/')

# %%
import xmltodict
import json

with open('./corpus_compressed/protocols/201617/prot-201617--33.xml', encoding='utf-8') as s:
    text = s.read()
#     print(text)
    res = xmltodict.parse(text,encoding='utf-8')
#     print(json.dumps(xmltodict.parse(text,encoding='utf-8')))

# %%
res['teiCorpus']['TEI']['text']['body']['div']['u']
