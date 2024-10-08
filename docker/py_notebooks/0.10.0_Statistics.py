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

# !{sys.executable} -m pip install nltk pyarrow openpyxl plotly bs4 lxml portion 
# !python --version

# %%
from IPython.display import Markdown
MD = lambda x: display(Markdown(x))

# %%
import sys
sys.path.append('../') # To allow for import of westac_statistics

from dataclasses import dataclass
from westac_statistics.case_one_gui import CaseOneGUI

# %load_ext autoreload
# %autoreload 2

import ipywidgets
import matplotlib.pyplot as plt
import os
from glob import glob
from pathlib import Path

import pandas as pd
import numpy as np
from IPython.display import display

import plotly.express as px

corpus_version_string = "0.10.0"
corpus_tag = f'v{corpus_version_string}'
output_path = f'./output/{corpus_version_string}/'
Path(output_path).mkdir(parents=True, exist_ok=True)

def scroll_display(who: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(HTML("<div style='height: 400px; overflow: auto; width: fit-content'>" +
                 who.to_html() +
                 "</div>"))


# %%
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %%
# %%time
from importlib import reload
from westac_statistics import corpus_parser,  case_one_gui
reload(corpus_parser)
from westac_statistics import corpus_parser
from westac_statistics.git_repository import GitRepo 

corpus_location = '../riksdagen-corpus/corpus/protocols/'

repo = GitRepo(corpus_location)
repo.check_and_clone('https://github.com/welfare-state-analytics/riksdagen-corpus.git',clone=False)

if repo.current_tag != corpus_tag:
    print('Incorrect corpus tag, switching...')
    repo.switch_to_tag(corpus_tag)

parser = corpus_parser.CorpusParser(corpus_location, f'{output_path}/corpus_db_{corpus_version_string}.feather')
parser.initialize(force_update=False) # Todo: Will this work on a new corpus? Possible this needs to be manually set to True once...

# %%
from importlib import reload
from westac_statistics import metadata_parser
reload(metadata_parser)
from westac_statistics import metadata_parser

metadata = metadata_parser.MetadataParser('../riksdagen-corpus/corpus/metadata/')
metadata.initialize()

# %%
metadata.add_affiliation(parser.speech_dataframe)

# %%
MD('Sanity check to ensure the material has been loaded.')
df = metadata.metadata['person']
df[df.wiki_id == 'Q109828321']

# %%
df = metadata.join_on(parser.speech_dataframe, metadata.metadata['party_abbreviation'].rename(columns={'party':'party_affiliation'}),column='party_affiliation') 
df.abbreviation = df.abbreviation.str.capitalize()
df = metadata.join_on(df, metadata.metadata['name'].rename(columns={'wiki_id':'who'}), column='who')
df = metadata.join_on(df, metadata.metadata['person'].rename(columns={'wiki_id':'who'}), column='who')
df = df.rename(columns={'abbreviation':'party_abbrev'})

df.loc[df[df.gender.isna()].index,'gender'] = 'unknown' # Set gender to 'unknown' instead of np.nan
df['year'] = df.date.apply(lambda x: x.year) # Add year (to make it easier to plot)

df.loc[df[df.party_abbrev.isna()].index, 'party_abbrev'] = '?' # Set 'nan' values to '?' instead
SPEECH_INDEX = df
SPEECH_INDEX.to_feather(f'output/{corpus_version_string}/speech_index_{corpus_version_string}.feather')

# %%
import importlib
import westac_statistics
importlib.reload(westac_statistics)
from westac_statistics.dataframe_optimizer import DataFrameOptimizer
print('Before:', SPEECH_INDEX.memory_usage(deep=True).sum() / 1024 ** 3)


SPEECH_INDEX = DataFrameOptimizer.optimize_dataframe(SPEECH_INDEX)
print('After:', SPEECH_INDEX.memory_usage(deep=True).sum() / 1024 ** 3)

# %%
MD('## Save empty speeches to csv file, using ~ (tilde) as separator.')
MD('Additionally, the top protocols with empty speeches are shown')
parser.empty_speeches[['file_name','protocol','who_intro_id','who_intro']].to_csv(f'output/{corpus_version_string}/empty_speeches.csv',sep='~')
parser.empty_speeches.protocol.value_counts().head(5)

# %%
MD('# Unknown party affiliation of known speakers per year')
speech = SPEECH_INDEX[SPEECH_INDEX.party_affiliation == 'unknown_missing']
px.line(speech.groupby(['year']).who.unique().apply(len))

# %%
# MD('# Temporary list of checking for members of parliament who has no party after 1920')
# df = pd.read_csv('https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus/dev/input/matching/member_of_parliament.csv')
# members_who_have_party_defined_at_least_once = df[~df.party.isna()].wiki_id.unique()
# # Strip people who have a defined party at least once!
# df = df[~df.wiki_id.isin(members_who_have_party_defined_at_least_once)]
# df.start = df.start.apply(metadata.convert_date)
# df.end = df.end.apply(metadata.convert_date)
# df['start_year'] = df.start.apply(lambda x: x.year)
# df['end_year'] = df.end.apply(lambda x: x.year)
# df2 = df[(df.start_year >= 1920) & (df.end_year >= 1920)].groupby('wiki_id').party.agg(list).apply(np.unique).to_frame()
# df3 = df2[df2.party.apply(len) == 1].party.apply(lambda x: x[0]).isna()
# unknown_ids = df3[df3 == True].index.values
# def remove_duplicates(df):
#     for col in df.columns:
#         df[col] = df[col].apply(lambda x: pd.Series(x).drop_duplicates().tolist() if isinstance(x, list) else x)
#         df[col] = df[col].apply(lambda x: np.nan if (isinstance(x, list) and not x) else x)
#     return df
# complete_df = remove_duplicates(df[df.wiki_id.isin(unknown_ids)].sort_values(by='wiki_id').groupby('wiki_id').agg(list))
# for col in complete_df.columns:
#     complete_df[col] = [x[0] if len(x) == 1 else x for x in complete_df[col]]
   
# complete_df.to_excel('input_member_of_parliament_by_missing_party.xlsx')
# complete_df

# %% [markdown]
# # Known people, who have no party in party_affiliation.csv

# %%
speech = SPEECH_INDEX[SPEECH_INDEX.party_affiliation == 'unknown_missing']
df = speech.value_counts('who')

df = pd.DataFrame(df)
df.columns = ['count']
df.to_csv(f'output/{corpus_version_string}/known_person_but_unknown_party.csv',encoding='utf-8-sig', sep=';')

display(df.head(10))

# %%
df = metadata.metadata['member_of_parliament']
MD(f'# Number of members of parliament missing either start or end date: {len(df[(df.end.isna() | df.start.isna())].wiki_id.unique())}')
df = df[~df.end.isna()]

df.end = df.end.apply(metadata.convert_date)
members_ending_after_1920 = df[df.end.apply(lambda x: x.year) >= 1920]
display(members_ending_after_1920)
who_ending_after_1920 = members_ending_after_1920.wiki_id.unique()
df = SPEECH_INDEX[(SPEECH_INDEX.party_affiliation == 'unknown_missing') | (SPEECH_INDEX.party_affiliation == 'unknown')]
# fancy join to include name...
MD('## People included in members_of_parliament who quit after 1920, and are also of unknown party affiliation')
metadata.join_on(df[df.who.isin(who_ending_after_1920)].who.value_counts().to_frame().reset_index(), SPEECH_INDEX[['who','name']].drop_duplicates(), column='who')


# %% [markdown]
# # Most popular dates for protocols... This should reasonably only be one protocol per date!
# As we can see, several dates are overrepresented.

# %%
df = SPEECH_INDEX.groupby('protocol').date.unique().apply(lambda x: x[0]).value_counts().head(30)
fig = px.bar(df)
fig.update_xaxes(type='category')
fig.show()

# %% [markdown]
# # Party colors, verify!

# %%
from westac_statistics import case_one_gui
df = pd.DataFrame(tuple(case_one_gui.CaseOneGUI.PARTY_COLORS_HEX.items()),columns=['party_abbreviation', 'color'])
def color_rows(row):
    color = row['color']
    return ['background-color: %s' % color]*len(row)

df.style.apply(color_rows, axis=1)

# %% [markdown]
# # Number of speeches by specific party affiliation (in decending order)

# %%
df = SPEECH_INDEX[['party_affiliation','party_abbrev']].groupby('party_affiliation').agg(list)
df['count'] = df['party_abbrev'].apply(len)
df['party_abbrev'] = df['party_abbrev'].apply(lambda x: np.unique(x))
df.sort_values(by='count',ascending=False)   

# %% [markdown]
# # Most common unknown speaker introductions (in decending order)

# %%
# %%time
df = pd.DataFrame(SPEECH_INDEX[(SPEECH_INDEX.who == 'unknown')])
print('Before:', df.memory_usage(deep=True).sum() / 1024 ** 3)
df['who_intro'] = df.who_intro.astype('str')
print('After:', df.memory_usage(deep=True).sum() / 1024 ** 3)

df.groupby('who_intro',sort=False).agg(list)


# %%
df = pd.DataFrame(SPEECH_INDEX[(SPEECH_INDEX.who == 'unknown')]).groupby('who_intro').agg(list)

def _get_year_range_string(years):
    min_year = np.min(years)
    max_year = np.max(years)
    if min_year == max_year:
        return f'{min_year}'
    return f'{min_year}-{max_year}'

df.year = df.year.apply(_get_year_range_string)
df['count'] = df.who.apply(len)
df = df[['year','count', 'file_name']].sort_values('count',ascending=False).reset_index()
df.file_name = df.file_name.apply(np.unique)
df.drop(index=df[df.who_intro == ''].index, axis=0, inplace=True) # Drop empty introduction before saving
df.to_csv(f'output/{corpus_version_string}/most_common_unknown_introductions_with_files_{corpus_version_string}.csv',encoding='utf-8-sig', sep=';')
df.head(30)

# %%
import plotly.express as px
df = metadata.metadata['person']
# display(df)
# px.bar(df.groupby('gender').count(), barmode='group')
fig = px.bar(df.groupby('gender').count()[['wiki_id', 'dead', 'riksdagen_id']],
       height=600,
       barmode='group', 
       labels={'gender':'Kön','value':'Antal','variable':'Datapunkt'},
            title='Översikt')
# print(fig)
translate = {'wiki_id':'Totalt antal', 'dead':'Avlidna', 'riksdagen_id':'Antal i riksdagen'}
fig.for_each_trace(lambda t: t.update(name = translate[t.name],
                                      legendgroup = translate[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, translate[t.name])
                                     )
                  )
fig

# %%
df = metadata.metadata['minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.join(metadata.metadata['person'], lsuffix='_left', rsuffix='_right')
df = df.drop(columns=[x for x in df.columns if x.endswith('_right')])
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['wiki_id_left'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['Total'] = df2.agg('sum',axis=1)
df2 = df2.sort_values(by='Total',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by total number and grouped by gender')

# %%

df = metadata.metadata['minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.join(metadata.metadata['person'], lsuffix='_left', rsuffix='_right')
df = df.drop(columns=[x for x in df.columns if x.endswith('_right')])
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['wiki_id_left'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['agg'] = df2.agg(lambda x: np.abs(np.diff(x)),axis=1)
# display(df2)
df2 = df2.sort_values(by='agg',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by largest difference first and grouped by gender')

# %%
df = metadata.metadata['minister']
unique_roles = sorted(df.role.unique())
# Most common minister roles... Changed most often?
# display(df.groupby(df.role).count().sort_values(by='person_id', ascending=False).head(10))

df = df.merge(metadata.metadata['person'], on='wiki_id')
df2 = pd.DataFrame(df.groupby(['role','gender']).nunique()['wiki_id'])
df2 = df2.unstack(level=1)
df2.columns = [y for x, y in df2.columns.to_flat_index()]
# px.bar(df2,barmode='group',height=600)
df2['agg'] = df2.agg(lambda x: np.abs(np.diff(x)),axis=1).fillna(-10000)
# display(df2)
df2 = df2.sort_values(by='agg',ascending=False)[['man', 'woman']]
px.bar(df2,barmode='group',height=600,title='Ministers, sorted by largest difference first and grouped by gender')

# %%
df = SPEECH_INDEX.groupby('file_name').who.nunique().sort_values(ascending=False)
df.name = 'Number of unique speakers'
display(px.histogram(df, title='Histogram of number of speakers per document'))

# %% [markdown]
# # Protocols with the largest number of tokens by unknowns

# %%
import plotly.graph_objects as go

df = SPEECH_INDEX[SPEECH_INDEX.who == 'unknown'].groupby(['file_name','gender']).agg({'n_tokens':'sum'}) #Change sum/len here!
df = df.join(df.groupby(['file_name']).agg({'n_tokens':'sum'}).rename(columns = {'n_tokens':'total_tokens'}),how='left').sort_values(by='total_tokens',ascending=False)
df = df.head(100).drop(columns='total_tokens').unstack(1)
df.columns = [x[1] for x in df.columns]
display(px.bar(df, barmode='stack',height=600))
# for a, b in SPEECH_INDEX[SPEECH_INDEX.who == 'unknown'].groupby(['file_name','gender']):
#     print(a)
#     print(b)


# %% [markdown]
# # Percentage of "uncertain" party assignments per assumed party

# %%
px.bar(SPEECH_INDEX.groupby('party_abbrev').agg({'party_affiliation_uncertain':'mean'}).sort_values(by='party_affiliation_uncertain',ascending=False))

# %% [markdown]
# # Total tokens per party
#  - Plot is sorted in ascending order from left (few) to right (many)

# %%
px.bar(SPEECH_INDEX.groupby('party_abbrev')['n_tokens'].sum().sort_values(),title='Total tokens uttered per party',height=600)

# %%
df = SPEECH_INDEX[['party_abbrev','year','n_tokens']]
df['decade'] = (df.year / 10).apply(int) * 10
res = df.groupby(['party_abbrev','decade']).n_tokens.sum().to_frame().reset_index()
MD('# Number of tokens per party per decade')
px.line(x=res.decade, y=res.n_tokens,color=res.party_abbrev)

# %% [markdown]
# # GUI Plots

# %%
import importlib
import westac_statistics.case_one_gui as cog
importlib.reload(cog)
PARTYS = SPEECH_INDEX.party_abbrev.unique().tolist()

df_year = SPEECH_INDEX
df_year['year'] = df_year.date.apply(lambda x: x.year)
df_year.party_abbrev = df_year.party_abbrev.str.upper()

guip = cog.CaseOneGUI(df_year, filter_key='party_abbrev', filter_sub_key='gender')\
    .setup(filter_values=[''] + PARTYS)
display(guip.layout())
guip.update()

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

# %% [markdown]
# # Number of unique speakers overall per year

# %%
px.line(SPEECH_INDEX.groupby('year').agg(**{'Number of unique speakers':('who','nunique')}))

# %%
from ipywidgets import Tab
from IPython.display import display, HTML, DisplayHandle
from itertools import chain


from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea

from plotly.subplots import make_subplots
def create_per_party_plots(base_df: pd.DataFrame, column_to_show='Tokens per member and year', title=None, share_y=True):
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
    if share_y:            
        # If plotly fixes yaxis_range, remove these and use correct method
        all_y = list(chain(*[x.y for x in fig.data]))
        min_y, max_y = np.nanmin(all_y), np.nanmax(all_y)
        fig.update_yaxes(range=[min_y,max_y])            
    return fig

output_left = Output()
output_right = Output()
tabbed_interface = Tab(children=[output_left, output_right])
tabbed_interface.set_title(0,'Average tokens per gender per party')
tabbed_interface.set_title(1,'Speakers per gender per party')
display(tabbed_interface)
with output_left:
    display(create_per_party_plots(SPEECH_INDEX, column_to_show='Tokens per member and year', title='Average spoken tokens per gender per party',share_y=False))
with output_right:
    display(create_per_party_plots(SPEECH_INDEX, column_to_show='Number of speakers', title='Number of unique speakers per party of different genders',share_y=False))

# %%
data = SPEECH_INDEX.groupby(['year','gender','party_abbrev']).agg(
    **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum')})
data['Tokens per member and year'] = np.round(data['Number of tokens (total)'] / data['Number of speakers'],2)
# Resetting index to ensure values exists for each row
data.reset_index().to_excel(f'output/{corpus_version_string}/tokens_per_year_grouped_by_gender_and_party.xlsx')
data = SPEECH_INDEX.groupby(['year','party_abbrev']).agg(
    **{'Number of speakers':('who','nunique'),'Number of speeches (total)':('year','count'), 'Number of tokens (total)':('n_tokens','sum')})
data['Tokens per member and year'] = np.round(data['Number of tokens (total)'] / data['Number of speakers'],2)
data.reset_index().to_excel(f'output/{corpus_version_string}/tokens_per_year_grouped_by_party.xlsx')

# %% [markdown]
# # Tokens from known vs unknown speakers per year

# %%
year_tokens_known = SPEECH_INDEX[SPEECH_INDEX.who!='unknown'].groupby('year').n_tokens.sum()
year_tokens_unknown = SPEECH_INDEX[SPEECH_INDEX.who=='unknown'].groupby('year').n_tokens.sum()
df = pd.DataFrame({'Known':year_tokens_known, 'Unknown':year_tokens_unknown})
df['UnknownPercentage'] = 100 * df.Unknown / (df.Known + df.Unknown)
display(px.bar(df, y='UnknownPercentage', title='Percentage of tokens spoken by unknown speakers per year'))
px.bar(df.UnknownPercentage.map(lambda x: (100-x) ).rename('Known part'), labels={'value':'Percent known'})

# %% [markdown]
# # Histogram of speech lengths
# ## Including a partial list of speeches that are very long (> 10000 tokens)

# %%
display(px.histogram(SPEECH_INDEX.n_tokens))
display(SPEECH_INDEX[SPEECH_INDEX.n_tokens > 10000].sort_values(by='n_tokens',ascending=False).head(10))

# %%
# with pd.option_context('display.max_colwidth', 1000):
#     df = SPEECH_INDEX[SPEECH_INDEX.who_intro.apply(len) > 50].who_intro.to_frame()
#     df['length'] = df.who_intro.str.len()
#     df = df.sort_values(by='length',ascending=False)
#     display(df.head(20))
speech_lengths = SPEECH_INDEX.who_intro.str.len()
long_speeches = speech_lengths > 200
data = speech_lengths[long_speeches]
df = SPEECH_INDEX[long_speeches].who_intro.str.len()
fig = px.histogram(df)
fig.update_xaxes(
    range=[200, 600],
    tickvals=list(range(200, 601, 100)),
    tickangle=45,
    title_text='Length'
)
MD('# Prominence of long introductions (>100 characters)')
display(fig)
display(px.line(SPEECH_INDEX[long_speeches].groupby('year').who.count()))
SPEECH_INDEX[long_speeches][['file_name','who_intro_id','year','who_intro']].sort_values(by='year').to_excel(f'output/{corpus_version_string}/long_introductions.xlsx')
                                                                                                          
                                                                                                          
                                                                                                          

# %%
import pandas as pd
import plotly.graph_objects as go

def plot_mean_std(df, year_col, value_col):
    # Group by year and calculate mean and standard deviation
    yearly_data = df.groupby(year_col)[value_col].agg(['mean', 'std'])

    # Create line plot for mean values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_data.index, 
        y=yearly_data['mean'],
        mode='lines',
        name='mean',
        line=dict(color='rgb(0,100,80)'),
        hovertemplate = 
            '<b>Year</b>: %{x}<br>'+
            '<b>Mean</b>: %{y:.2f}<br>'+
            '<b>Std Dev</b>: '+ yearly_data['std'].apply('{:.2f}'.format) +'<extra></extra>',
    ))

    # Add shaded area for standard deviation
    fig.add_trace(go.Scatter(
        x=yearly_data.index.tolist() + yearly_data.index.tolist()[::-1], 
        y=(yearly_data['mean'] + yearly_data['std']).tolist() + (yearly_data['mean'] - yearly_data['std']).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title='Mean and Standard Deviation of Values per Year',
        xaxis_title=year_col,
        yaxis_title=value_col,
    )

    return fig

def plot_quantiles(df, year_col, value_col, lower_quantile=0.25, upper_quantile=0.75):
    # Group by year and calculate quantiles
    yearly_data = df.groupby(year_col)[value_col].agg([lambda x: x.quantile(lower_quantile), 'median', lambda x: x.quantile(upper_quantile)])

    yearly_data.columns = ['lower_quantile', 'median', 'upper_quantile']

    # Create line plot for median values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_data.index, 
        y=yearly_data['median'],
        mode='lines',
        name='median',
        line=dict(color='rgb(0,100,80)'),
        hovertemplate = 
        '<b>Year</b>: %{x}<br>'+
        '<b>Median</b>: %{y:.2f}<br>'+
        '<b>Lower Quantile</b>: '+ yearly_data['lower_quantile'].apply('{:.2f}'.format) +'<br>'+
        '<b>Upper Quantile</b>: '+ yearly_data['upper_quantile'].apply('{:.2f}'.format) +'<extra></extra>',
    ))

    # Add shaded area for quantiles
    fig.add_trace(go.Scatter(
        x=yearly_data.index.tolist() + yearly_data.index.tolist()[::-1], 
        y=yearly_data['upper_quantile'].tolist() + yearly_data['lower_quantile'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title=f'Quantiles of Values per Year, lower quantile = {lower_quantile}, upper quantile = {upper_quantile}',
        xaxis_title=year_col,
        yaxis_title=value_col,

    )

    return fig


df = SPEECH_INDEX[['who_intro','year']]
df['Introduction_length'] = df.who_intro.str.len()
MD('# Length of introduction by year, including standard deviation')
display(plot_mean_std(df, 'year', 'Introduction_length'))
display(plot_quantiles(df, 'year', 'Introduction_length', lower_quantile=0.25, upper_quantile=0.75))
display(plot_quantiles(df, 'year', 'Introduction_length', lower_quantile=0.1, upper_quantile=0.9))


# %%

# %% [markdown]
# # Average age of speakers per party and gender

# %%
df = SPEECH_INDEX
df['approx_age'] = (df.date - df.born).apply(lambda x: x.days / 365.2425) # Estimated number of days per year

print('------ People speaking after age 100 (first 20)')
print(df[df.approx_age > 100].who.unique())


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
                        shared_xaxes=True, shared_yaxes=True)
    fig.update_layout(hovermode='x')
    #                     subplot_titles=party_list)
    for i, party in enumerate(party_list):
        d = data[data.index.get_level_values('party_abbrev') == party][column_to_show].unstack(level=0).T
        d.columns = [g for g, p in d.columns]
        px_fig = px.line(d)
        for trace in px_fig.data:
            fig.append_trace(trace, row=i+1, col=1)

    fig.update_layout(height=2000, title_text=title)

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

fig = create_per_party_ages(df, column_to_show='Average age', title='Average age of speaker per party')
fig.update_yaxes(range=[30,80])
display(fig)

# %% [markdown]
# # People who are not unknown but have party unknown
# ## Likely this includes parties with invalid party->party_abbrev listings!

# %%
df = SPEECH_INDEX[SPEECH_INDEX.who != 'unknown']

df = df[df.party_abbrev == '?'].groupby('who').agg(list)
df['count'] = df.party_affiliation.apply(len)
df.sort_values(by='count', ascending=False)
__tst = list()

def flatten_and_unique(input_list):
    result_set = set()

    def helper(sub_list):
        if not isinstance(sub_list, list):
            result_set.add(sub_list)
            return
        
        for item in sub_list:
            if isinstance(item, list):
                helper(item)
            else:
                result_set.add(item)

    helper(input_list)
    lst = list(result_set)
    if len(lst) == 1:
        return lst[0]
    return lst


def move_column_in_position(df, column, position):
    col = df.pop(column)
    df.insert(position, column, col)

for col in df.columns:
        
    df[col] = df[col].apply(lambda x: flatten_and_unique(x))
    # df[col] = df[col].apply(lambda x: x[0] if len(x) == 1 else x)
df['Year_range'] = df.year.apply(lambda x: [np.min(x), np.max(x)]).apply(np.diff)    
move_column_in_position(df, 'count',0)
df.sort_values(by='count', ascending=False).head(5)

# %%
### import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML, DisplayHandle


from ipywidgets import Dropdown, VBox, HBox, Label, Output, Text, Textarea

selected_documents = []

df = SPEECH_INDEX.set_index('u_id')['n_tokens']
# df = SPEECH_INDEX.groupby(['protocol_name'])['n_tokens'].sum()


fig = px.histogram(df, x="n_tokens", nbins=100)
fig.update_layout(bargap=0.2,
                  title='Histogram of number of tokens per speech', 
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
        display(SPEECH_INDEX[SPEECH_INDEX.u_id.isin(set(selected_documents))])
    

fig_widget = go.FigureWidget(fig.data, fig.layout)
# print(dir(fig))
fig_widget.data[0].on_selection(do_things)
fig_widget.data[0].on_click(do_things)

display(HBox([fig_widget, VBox([Label(f'Selected speeches (out of {len(df)})\n'), text_output])]))
display(df_output)


# %% [markdown]
# # TTR AND RELATED CALCULATIONS HAVE MOVED

# %% [markdown]
# # Save speech count over time to file

# %%
def get_excel_writer_args(excel_file):
    if Path(excel_file).exists():
        extra_args = {'mode':'a', 'if_sheet_exists':'replace'}
    else:
        extra_args = {'mode':'w'}
    return extra_args

data = SPEECH_INDEX.groupby(['year'])['u_id'].count().rename('count')
# data.index = [year for tokens, year in data.index]
excel_file = f'./output/{corpus_version_string}/Speech_count_over_time.xlsx'
args = get_excel_writer_args(excel_file)
with pd.ExcelWriter(excel_file, engine='openpyxl', **args) as writer:
    data.to_excel(writer,sheet_name='All speeches')
plot = px.line(data,title='Speech count over time', labels={'index':'Year','value':'count'},height=600)
display(plot.update_layout(hovermode='x'))

MINIMUM_TOKENS = 5
data = SPEECH_INDEX[SPEECH_INDEX.n_tokens > MINIMUM_TOKENS].groupby(['year'])['u_id'].count().rename('count')
args = get_excel_writer_args(excel_file)
# data.index = [year for tokens, year in data.index]
with pd.ExcelWriter(excel_file, engine='openpyxl', **args) as writer:
    data.to_excel(writer,sheet_name=f'Speeches longer than {MINIMUM_TOKENS} tokens')
plot = px.line(data,title=f'Speech count over time, only speeches with > {MINIMUM_TOKENS} tokens', labels={'index':'Year','value':'count'},height=600)
plot.update_layout(hovermode='x')

# %% [markdown]
# # Save statistics on mean number of tokens spoken per gender, party and person to file

# %%
SPEECH_INDEX.groupby(['year','party_abbrev']).n_tokens.mean().unstack().to_excel(f'./output/{corpus_version_string}/Average tokens per year per party.xlsx')
SPEECH_INDEX.groupby(['year','gender']).n_tokens.mean().unstack().to_excel(f'./output/{corpus_version_string}/Average tokens per year per gender.xlsx')
SPEECH_INDEX.groupby(['year']).n_tokens.mean().to_excel(f'./output/{corpus_version_string}/Average tokens per year per person.xlsx')

# %%
# Speeches over time
SPEECH_INDEX.groupby(['year']).u_id.count().to_excel(f'./output/{corpus_version_string}/Speeches per year over time.xlsx')

# %%
df = metadata.metadata['party_affiliation']
df = metadata.join_on(df, pd.DataFrame(df.groupby('wiki_id').party.count()).reset_index().rename(columns={'party':'party_count'}), column='wiki_id')
ids = []
for name, group in df[(df.party_count >= 2)].groupby('wiki_id'):
    if len(group.start_dt.unique()) == 1 and len(group.end_dt.unique()) == 1:
        ids.extend(group.wiki_id.unique())
undefined_timespan_ids = set(ids)

# %%
SPEECH_INDEX[SPEECH_INDEX.who.isin(undefined_timespan_ids)].groupby('year').who.value_counts().unstack().to_excel(f'./output/{corpus_version_string}/Undefined party by year.xlsx')

# %% [markdown]
# # Bar plot showing % of people having uncertain affiliations, per year

# %%
px.bar(SPEECH_INDEX.groupby('year').party_affiliation_uncertain.mean() * 100, title='Percentage of people having uncertain affiliation per year', labels={'value':'Percent'})

# %% [markdown]
# # Find people with overlapping affiliations in the metadata

# %%
df = metadata.metadata['party_affiliation'].copy()
df.loc[df[df.end_dt.isnull()].index, 'end_dt'] =np.datetime64('today')
first_date = df.start_dt.sort_values().values[0]
df.loc[df[df.start_dt.isnull()].index, 'start_dt'] = first_date
df2 = df
df2['overlap'] = (df2.groupby('wiki_id')
                       .apply(lambda x: (x['end_dt'].shift() - x['start_dt']) > pd.Timedelta(0))
                       .reset_index(level=0, drop=True))

def different_party_overlap(dataframe):
    found_overlap = np.where(dataframe.overlap)[0]
    for idx in found_overlap:
        if dataframe.iloc[[idx-1,idx]].party.nunique() > 1:
            return True
    return False

total_dataframes = []
for id, count in df2[df2.overlap == True].wiki_id.value_counts().items():
    tmp = df2[df2.wiki_id == id]
    if tmp.party.nunique() == 1:
        continue
    # if not different_party_overlap(tmp):
    #     continue
    total_dataframes.append(df2[df2.wiki_id == id])
    # display(df2[df2.wiki_id == id])
overlap_affiliations = pd.concat(total_dataframes)
overlap_affiliations.to_excel(f'./output/{corpus_version_string}/metadata_party_affiliations_overlap.xlsx')
overlapping_ids = set(overlap_affiliations.wiki_id.values)
SPEECH_INDEX[SPEECH_INDEX.who.isin(overlapping_ids)].who.value_counts().head(20)

# %%
# def different_party_overlap(dataframe):
import portion as P    
from functools import reduce
from tqdm.auto import tqdm
# dataframe = df2[df2.wiki_id == 'Q271468']
# # display(dataframe)
# # dataframe = dataframe.drop(index = 1068)
# # dataframe = dataframe.drop(index = 1067)
# # dataframe.loc[9703,'start_dt'] = pd.Timestamp('2003')
# display(dataframe)

# different_party_overlap(dataframe)

# intervals = P.IntervalDict({P.closed(row.start_dt,row.end_dt):row.party for idx, row in dataframe.iterrows()})
# for x in intervals.items():
#     print(x)
def check_overlap_for_user(id, user_df):
    if user_df.party.nunique() < 2:
        return []
    
    group_intervals = dict()
    for name, group in user_df.groupby('party'):
        group_intervals[name] = reduce(lambda a,b: a|b, [P.open(row.start_dt,row.end_dt) for idx, row in group.iterrows()])

    parties = list(group_intervals.keys())
    overlap = []
    for i in range(len(parties)):
        for j in range(i+1, len(parties)):
            # print(f'Testing {parties[i]} vs {parties[j]}.')
            a = group_intervals[parties[i]]
            b = group_intervals[parties[j]]
            
            
            if a.overlaps(b):
                intersect = a.intersection(b)
                overlap.append(
                    {'wiki_id':id,
                     'Overlap start':intersect.lower,
                     'Overlap end':intersect.upper,
                     'Party 1':parties[i],
                     'Party 2':parties[j]
                    })
                     
                # display(user_df)
                
                # print(f'Overlap between {intersect.lower} to {intersect.upper}!')
                # print(f'Parties: {parties[i]} and {parties[j]}.')
    return overlap
            
# df = metadata.metadata['party_affiliation']    
all_overlap = []
for id, group in tqdm(df.groupby('wiki_id')):
    all_overlap.extend(check_overlap_for_user(id, group))
df2 = pd.DataFrame(all_overlap)

# %%
MD('# Speakers that have overlapping party affiliations, i.e. belong to more than one party at once')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    res = df2.sort_values(by='Overlap start').reset_index(drop=True)
    res['Overlap start'] = res['Overlap start'].apply(lambda x: x.strftime('%Y-%m-%d'))
    res['Overlap end'] = res['Overlap end'].apply(lambda x: x.strftime('%Y-%m-%d'))
    display(res)
    
    res.to_excel(f'./output/{corpus_version_string}/metadata_party_affiliations_overlap_with_dates.xlsx')

# %%

# %%
import plotly.figure_factory as ff

def visualize_overlap(df, wiki_id):
    # Filter DataFrame by wiki_id
    df_filtered = df[df.wiki_id == wiki_id]
    print('Earliest: ', df_filtered['start_dt'].min())
    display(df_filtered['start_dt'])

    # Ensure start_dt and end_dt are not NaT
    df_filtered['start_dt'] = df_filtered['start_dt'].fillna(df_filtered['start_dt'].min())
    df_filtered['end_dt'] = df_filtered['end_dt'].fillna(df_filtered['end_dt'].max())

    # Display filtered DataFrame
    display(df_filtered)

    # Create Gantt chart
    df_gantt = df_filtered[['party','start_dt','end_dt']]
    df_gantt.columns = ['Task','Start','Finish']
    fig = ff.create_gantt(df_gantt)

    return fig

fig = visualize_overlap(metadata.metadata['party_affiliation'], 'Q3369258')
fig.show()


# %% [markdown]
#
# # Generate "uncertain per decade" spreadsheet and sort according to when the speaker has spoken
# TODO: There absolutely must be a better way to sort the speakers according to when they were speaking...

# %%
df = SPEECH_INDEX[SPEECH_INDEX.party_affiliation_uncertain]
df['decade'] = df.year.apply(lambda x: int(x/10) * 10)
df

# px.imshow(pd.DataFrame(df.groupby('decade').who.value_counts().unstack()))
df = pd.DataFrame(df.groupby('decade').who.value_counts().unstack()).T
average_decade = []
for i in range(len(df)):
    count = 0
    tot = 0
    for dec, val in df.iloc[i].items():
        if np.isnan(val):
            continue
        count += val
        tot += dec * val
    average_decade.append(tot / count)
df['average_decade']= average_decade
df = df.sort_values('average_decade')

df.drop(columns='average_decade').to_excel(f'./output/{corpus_version_string}/uncertain_per_decade.xlsx')

# %%
import seaborn as sns

# %%
# %%time
# # !sudo pip install swifter

df = SPEECH_INDEX
import hashlib

df['hash'] = df.text.apply(lambda x: hashlib.md5(x[0].encode('utf-8')).hexdigest())
print('Done with hash...')

new_df_list = []
for h, repeats in df.hash.value_counts().items():
    if repeats < 10:
        break
    this_df = df[df.hash == h]
    new_df_list.append({'hash':this_df.iloc[0].hash,
                        'repeats':repeats,
                        'text':this_df.iloc[0].text[0]
                       })
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'max_colwidth', 200):
    display(pd.DataFrame(new_df_list).head(20))    


# %% [markdown]
# Hej Tomas! Johan och jag har diskuterat några funktioner som vi skulle vilja att du tar fram gällande riksdagstalens längd.
#
#  
#
# Vi vill se (A.) hur många tal av en viss längd förändras över tid, och (B.) den procentuella mängden tal med en viss tallängd jämfört med alla tal under en viss tidsperiod.
#
#  
#
# Vi skulle kunna dela in alla tal i tre längdkategorier: tal med upp till 200 ord, tal med 201–500 ord samt tal med mer än 500 ord. Det vi alltså vill ha då är (A.) en trendgraf som visar hur många tal per år som är mellan 1–200 ord, liksom för 201–500 ord och 501+ ord. Vi vill också (B.) se hur många procent som en viss tallängdkategori utgör av det totala antalet tal per decennium (blir nog lättast).

# %%
# import pandas as pd
# SPEECH_INDEX = pd.read_feather(f'output/{corpus_version_string}/speech_index_{corpus_version_string}.feather')

# %%
class SpeechLengthOverTime:
    def __init__(self, df, output_path):
        self.df = df[df.n_tokens < 10000].copy()
        self.output_path = output_path
        self.token_ranges = [(20,200), (201,500), (501, 1000), (1001, np.inf)]
        self.convert_table = {'1001-inf':4, '20-200':0, '201-500':2, '501-1000':3}

    def preprocess(self):
        self.df['decade'] = self.df.year.apply(lambda x: int(x/10) * 10)
        self.df['token_range'] = self.df.n_tokens.apply(
            lambda x: next(iter([f'{a}-{b}' for a,b in self.token_ranges if ((x >= a) & (x <= b))]),'None')
        )
        self.df = self.df.drop(self.df[self.df.token_range == 'None'].index)

    def analyze(self):
        decade = self.df.groupby(['year', 'token_range']).count()['who'].rename('count')
        df2 = decade.reset_index()
        df2.to_excel(f'{self.output_path}/speech_count_per_token_count_per_year.xlsx')
        df2['order'] = df2.token_range.apply(lambda x: self.convert_table[x])
        df2 = df2.sort_values(by=['year','order'])
        display(px.line(df2, x='year', y='count', color='token_range'))

        df3 = self.df.groupby(['year','token_range']).count()['who'].rename('count').reset_index()
        df3['normalized_count'] = df3['count'] / df3.groupby('year')['count'].transform('sum')
        df3.to_excel(f'{self.output_path}/speech_normalized_count_per_token_range_per_year.xlsx')
        df3['order'] = df3.token_range.apply(lambda x: self.convert_table[x])
        df3 = df3.sort_values(by='order')
        display(px.area(df3, x='year', y='normalized_count', color='token_range'))

        df4 = self.df.groupby(['year', 'token_range', 'gender', 'party_abbrev']).count()['who'].rename('count').reset_index()
        df4['normalized_count'] = df4['count'] / df4.groupby(['year', 'gender', 'party_abbrev'])['count'].transform('sum')
        df4.to_excel(f'{self.output_path}/speech_normalized_count_per_token_range_per_year_party_gender.xlsx')

        df5 = self.df.groupby(['year','party_abbrev', 'token_range']).count()['who'].rename('count').reset_index()
        df5['normalized_count'] = df5['count'] / df5.groupby(['year','party_abbrev'])['count'].transform('sum')
        df5.to_excel(f'{self.output_path}/speech_normalized_count_per_token_range_per_year_and_party.xlsx')

    def run(self):
        self.preprocess()
        self.analyze()

analysis = SpeechLengthOverTime(SPEECH_INDEX, output_path)
analysis.run()

# %%

# %%
MD('# BELOW THIS, WE HAVE TEMPORARY TESTS; IGNORE')

# %%
#  ta fram hur många unika MPs för ett visst parti det fanns för varje årtionde? Isf skulle det väl vara ett någorlunda rimligt sätt att normalisera unika ord?
df = metadata.metadata['member_of_parliament'].copy()
df.start = df.start.apply(metadata.convert_date)
df.end = df.end.apply(metadata.convert_date)
year_to_range = lambda year: (pd.to_datetime(f'{year}-01-01 00:00:00'), pd.to_datetime(f'{year}-12-31 23:59:59'))
decade_to_range = lambda year: (pd.to_datetime(f'{year//10*10}-01-01 00:00:00'), pd.to_datetime(f'{year//10*10+9}-12-31 23:59:59'))

start_date,  end_date = decade_to_range(2001)

filtered_df = df[((df['start'] <= end_date) & (df['end'] >= start_date))]


def check_values(df, column, values):
    # Get the unique values in 'column' that are in the 'values' list
    found_values = df[df[column].isin(values)]
    

    # Convert the found values to a set
    found_values_set = set(found_values[column].values)

    # Convert the original values to a set
    values_set = set(values)

    # Find the values that were not found
    not_found_values = values_set - found_values_set



    # Print a tally of how many values were found vs not found
    print(f"Number of values found: {len(found_values_set)}")
    print(f"Number of values not found: {len(not_found_values)}")

    # Print the values that were not found
    if not_found_values:
        print(f"The following values were not found: {', '.join(not_found_values)}")
        
    return found_values

from IPython.display import HTML

df_html = filtered_df.sort_values(by='start').to_html()


# HTML(f'<div style="max-height: 1000px; overflow: auto;">{df_html}</div>')
check_values(metadata.metadata['party_affiliation'], 'wiki_id', filtered_df.wiki_id.values).party.value_counts()

# %%
tst1 = check_values(metadata.metadata['party_affiliation'], 'wiki_id', filtered_df.wiki_id.values).groupby('wiki_id')['party'].agg('unique').to_frame()
tst1
tst1['party_count'] = tst1.party.apply(len)
for who in tst1[tst1.party_count > 1].index.values:
    closest, uncertain, message = metadata.get_affiliation_from_dates(who, [pd.to_datetime('2006-01-01')])[0]
    print(who, ':', closest, ' - ', uncertain, ' - ', message)
    tst1.loc[who].party = [closest]


tst1.party = tst1.party.apply(lambda x: x[0])
tst1.drop(columns='party_count').party.value_counts()

# %%
SPEECH_INDEX['decade'] = SPEECH_INDEX.year.apply(lambda x: int(x / 10) * 10)
df = SPEECH_INDEX[SPEECH_INDEX.decade == 2010].groupby('who').agg(list)
df.party_affiliation = df.party_affiliation.apply(np.unique)
df.party_affiliation.value_counts()

# %%
tst1.party.value_counts()

# %%
check_values(metadata.metadata['party_affiliation'], 'wiki_id', filtered_df.wiki_id.values).groupby('wiki_id').party.agg('unique')

# %%
check_values(metadata.metadata['party_affiliation'], 'wiki_id', filtered_df.wiki_id.values).wiki_id.value_counts()

# %%



# %%

# %%

# %%

# %%

# %%

# %%

# %%
metadata.metadata['party_affiliation'].head(5).to_csv()

# %%
df = SPEECH_INDEX[SPEECH_INDEX.who_intro == 'Hans excellens herr statsministern ERLANDER:']
# df[df.who == 'unknown'].protocol.value_counts().head(5)
df.who.value_counts()

# %%
df = SPEECH_INDEX.copy()
df.who_intro = df.who_intro.str.lower()
df = df[~df.who_intro.str.contains('talmannen')]
df = df[~df.who_intro.str.contains('talman')]
df = df[~((df.who_intro.str.contains('herr') )& (df.who_intro.str.split().apply(len) < 3))]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(pd.DataFrame(df.groupby('who_intro').who.nunique().sort_values(ascending=False).items(), columns=['Intro','Count']).head(100))

# %%
MD('##Test of finding words in metadata')

for name, df in metadata.find_words_in_metadata(words = ['Robert','Johansson']).items():
    print(name)
    display(df)

# %%
# Vi kör på 0.10.x
# Ny data för spann (antal tokens)
# LIX-värdet 
# Långa ord, hur många tecken ett ord innehåller (blir excel-dokument)
# TTR, 4 olika, för decennium och parti
# Fraser, när jag tog fram 7-grams (bara 7-grams vi är intresserad av) så vill vi ändra så att ord med bara ett tecken tas bort.
#   För just denna metod vill vi ha med alla ord!
# Samma kolumner som sist med TF-IDF osv osv
# VI MÅSTE UNDERSÖKA TF-IDF ännu en gång'
### 1920
###### Även en variant som är rankad helt efter Total Count och inte TF-IDF
###### I den absoluta listan ska inte stoppordlistan användas!

# %%
#Gällande talens medellängd över tid – kan du ta fram medellängden för tal fördelat på S-män/S-kvinnor, M-män/M-kvinnor, V-män/V-kvinnor… osv?
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def plot_average_length(df, output_file = None):
    # Ensure the date is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Extract the year from the date
    df['year'] = df['date'].dt.year

    # Group by year, party_abbrev, and gender, then calculate the mean n_tokens
    grouped = df.groupby(['year', 'party_abbrev', 'gender'])['n_tokens'].mean().reset_index()

    # Get the unique parties
    parties = grouped['party_abbrev'].drop_duplicates()
    
    # Find the maximum value across all your data
    max_value = grouped['n_tokens'].max()
    
    # Get the unique genders and assign a distinct color to each
    genders = df.gender.unique()
    colors = sns.color_palette('colorblind', len(genders)).as_hex()  # generates a list of distinct colors

    # Create a dictionary mapping each gender to a color
    gender_color_dict = dict(zip(genders, colors))

    # Create a subplot for each party
    fig = make_subplots(rows=len(parties), cols=1, shared_xaxes=True, vertical_spacing=0.01)

    # For each party, create a line plot for each gender
    for i, party in enumerate(parties, start=1):
        for gender in genders:
            data = grouped[(grouped['party_abbrev'] == party) & (grouped['gender'] == gender)]
            if not data.empty:
                fig.add_trace(go.Scatter(x=data['year'], y=data['n_tokens'], mode='lines', name=f"{gender}", line=dict(color=gender_color_dict[gender]), showlegend=(i==1)), row=i, col=1)
        fig.update_yaxes(title_text=party, row=i, col=1, range=[0, max_value])

    fig.update_layout(height=100*len(parties), title_text='Average Number of Tokens Over Time by Party and Gender', hovermode='x unified')
    fig.update_xaxes(matches='x')
    fig.show()

    if output_file:
        grouped.to_excel(output_file)
    
plot_average_length(SPEECH_INDEX, output_file=f'{output_path}/average_tokens_by_party_and_gender.xlsx')


# %%
# %%time
import shutil

def create_text_files_and_zip(df, base_directory, start_date=None, end_date=None):
    # Convert dates to datetime if they are not None
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    # Filter DataFrame based on date range if start_date and end_date are not None
    if start_date is not None and end_date is not None:
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df.loc[mask]

    for name, group in df.groupby('party_abbrev'):
        # Create a directory for each name
        directory = os.path.join(base_directory, name)
        os.makedirs(directory, exist_ok=True)

        # For each row in the group, create a text file
        for index, row in group.iterrows():
            # Create a filename based on the date and 'who' column
            filename = f"{row['date'].strftime('%Y-%m-%d')}_{row['who']}_{row['who_intro']}.txt"
            filepath = os.path.join(directory, filename)

            # Write the contents of the 'text' column to the file
            with open(filepath, 'w') as file:
                file.write(str('\n\n'.join(row['text'])))

    # Create a zip file of the base directory
    shutil.make_archive(base_directory, 'zip', base_directory)


# Fredrik wanted some data for test purposes, so this writes some text files from the corpus. 
# If need be, recreate them later by oncommenting
    
# BASE_DIRECTORY = os.path.join(output_path, 'speeches_by_party')
# start_date = '2021-09-10'
# end_date = '2022-09-11'
# create_text_files_and_zip(SPEECH_INDEX, BASE_DIRECTORY, start_date, end_date)

# %%
# %%time
import re
import plotly.io as pio
import plotly.express as px
from tqdm.auto import tqdm
import pandas as pd
from multiprocessing import Pool

class TextAnalysis:
    def __init__(self, df, threads=1):
        self.df = df.copy()
        self.yearly_data = None
        self.fig = None
        self.regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        self.threads = threads
        tqdm.pandas()

    def count_sentences(self, text):
        
        pattern = re.compile(self.regex)
        merged_text = ' '.join(text)
        comma_count = merged_text.count(',')
        dash_count = merged_text.count(' - ')
        semicolon_count = merged_text.count(';')
        sentences = pattern.split(merged_text)
        sentence_count = len(sentences)
        return sentence_count, comma_count, dash_count, semicolon_count

    def threaded_count_sentences(self):
        global worker_fun

        def worker_fun(texts):
            worker_result = []
            for i, text in enumerate(self.df.text.values[texts]):
                merged_text = ' '.join(text)
                pattern = re.compile(self.regex)                
                comma_count = merged_text.count(',')
                dash_count = merged_text.count(' - ')
                semicolon_count = merged_text.count(';')
                sentences = pattern.split(merged_text)
                sentence_count = len(sentences)                
                worker_result.append((sentence_count, comma_count, dash_count, semicolon_count))
            return worker_result

        chunks = np.array_split(range(len(self.df)), self.threads)
        with Pool(self.threads) as pool:
            all_results = pool.map(worker_fun, chunks)

        del worker_fun

        # Flatten all_results into a single list of tuples
        merged_results = [item for sublist in all_results for item in sublist]

        return merged_results

    def calculate_avg_tokens(self):
        if self.threads > 1:
            counts = pd.Series(self.threaded_count_sentences(), index=self.df.index)
        else:
            counts = self.df.text.progress_apply(self.count_sentences)
        self.df[['sentence_count', 'comma_count', 'dash_count', 'semicolon_count']] = pd.DataFrame(counts.tolist(), index=self.df.index)
        self.yearly_data = self.df.groupby('year').agg({'n_tokens': 'sum', 'sentence_count': 'sum', 'comma_count':'sum', 'dash_count':'sum', 'semicolon_count':'sum'})
        self.yearly_data['avg_tokens_per_sentence'] = self.yearly_data['n_tokens'] / self.yearly_data['sentence_count']
        self.yearly_data['avg_tokens_per_comma'] = self.yearly_data['n_tokens'] / self.yearly_data['comma_count']
        self.yearly_data['avg_tokens_per_dash'] = self.yearly_data['n_tokens'] / self.yearly_data['dash_count']
        self.yearly_data['avg_tokens_per_semicolon'] = self.yearly_data['n_tokens'] / self.yearly_data['semicolon_count']
        self.yearly_data['total_dividers'] = self.yearly_data['comma_count'] + self.yearly_data['dash_count'] + self.yearly_data['semicolon_count']
        self.yearly_data['avg_tokens_per_divider'] = self.yearly_data['n_tokens'] / self.yearly_data['total_dividers']
        

    @property
    def figure(self):
        if self.yearly_data is None:
            self.calculate_avg_tokens()
        self.fig = px.line(self.yearly_data, title='Average Tokens per Sentence, Comma, Dash, and Semicolon Over Years')
        self.fig.update_yaxes(title_text='Count')
        return self.fig

    def save_figure(self, output_path, corpus_tag):
        if self.fig is None:
            self.figure
        pio.write_html(self.fig, os.path.join(output_path,f'{corpus_tag}_average_tokens_per_sentence_comma_dash_semicolon.html'))


analysis = TextAnalysis(SPEECH_INDEX, threads=16)
analysis.calculate_avg_tokens()
fig = analysis.figure
analysis.save_figure(output_path, corpus_tag)
analysis.yearly_data.to_excel(os.path.join(output_path,f'{corpus_tag}_sentence_analysis_mt.xlsx'))
fig


# %%
def get_speeches_per_party_and_decade(df, filename=None):
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Create a new column for the decade
    df_copy['decade'] = (df_copy['date'].dt.year//10)*10

    # Pivot the dataframe to get the count of party_abbreviation by decade
    result_df = df_copy.groupby(['decade', 'party_abbrev']).size().unstack().fillna(0)

    # If a filename is provided, save the result to an Excel file
    if filename:
        result_df.to_excel(filename)

    return result_df

get_speeches_per_party_and_decade(SPEECH_INDEX, os.path.join(output_path, f'{corpus_version_string}_no_speeches_per_party_and_decade.xlsx'))

# %%

# %%
