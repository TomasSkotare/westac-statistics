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

corpus_version_string = "0.9.0"
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

parser = corpus_parser.CorpusParser('../riksdagen-corpus/corpus/protocols/', f'{output_path}/corpus_db_{corpus_version_string}.feather')
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
MD('# Unknown party affiliation of known speakers per year')
speech = SPEECH_INDEX[SPEECH_INDEX.party_affiliation == 'unknown_missing']
px.line(speech.groupby(['year']).who.unique().apply(len))

# %%
MD('# Temporary list of checking for members of parliament who has no party after 1920')
df = pd.read_csv('https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus/dev/input/matching/member_of_parliament.csv')
members_who_have_party_defined_at_least_once = df[~df.party.isna()].wiki_id.unique()
# Strip people who have a defined party at least once!
df = df[~df.wiki_id.isin(members_who_have_party_defined_at_least_once)]
df.start = df.start.apply(metadata.convert_date)
df.end = df.end.apply(metadata.convert_date)
df['start_year'] = df.start.apply(lambda x: x.year)
df['end_year'] = df.end.apply(lambda x: x.year)
df2 = df[(df.start_year >= 1920) & (df.end_year >= 1920)].groupby('wiki_id').party.agg(list).apply(np.unique).to_frame()
df3 = df2[df2.party.apply(len) == 1].party.apply(lambda x: x[0]).isna()
unknown_ids = df3[df3 == True].index.values
def remove_duplicates(df):
    for col in df.columns:
        df[col] = df[col].apply(lambda x: pd.Series(x).drop_duplicates().tolist() if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: np.nan if (isinstance(x, list) and not x) else x)
    return df
complete_df = remove_duplicates(df[df.wiki_id.isin(unknown_ids)].sort_values(by='wiki_id').groupby('wiki_id').agg(list))
for col in complete_df.columns:
    complete_df[col] = [x[0] if len(x) == 1 else x for x in complete_df[col]]
   
complete_df.to_excel('input_member_of_parliament_by_missing_party.xlsx')
complete_df

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
display(SPEECH_INDEX[SPEECH_INDEX.n_tokens > 10000].sort_values(by='n_tokens',ascending=False).head(5))

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
MD('# Overlap visualization... Remove?')
display(df2[df2.wiki_id == 'Q5556026'])
df3 = metadata.metadata['party_affiliation']
display(df3[df3.wiki_id == 'Q5556026'])
df4 = df3[df3.wiki_id == 'Q5556026'][['party','start_dt','end_dt']]
df4.columns = ['Task','Start','Finish']
ff.create_gantt(df4)

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
df = SPEECH_INDEX[SPEECH_INDEX.n_tokens < 10000]
df['decade'] = df.year.apply(lambda x: int(x/10) * 10)

token_ranges = [(20,200), (201,500), (501, 1000), (1001, np.inf)]
convert_table = {'1001-inf':4, '20-200':0, '201-500':2, '501-1000':3}

df['token_range'] = df.n_tokens.apply(lambda x: next(iter([f'{a}-{b}' for a,b in token_ranges if ((x >= a) & (x <= b))]),'None'))
df = df.drop(df[df.token_range == 'None'].index)


decade = df.groupby(['year','token_range']).count()['who'].rename('count')

df2 = decade.reset_index()
# px.bar(df2, x='year', y='count', color='token_range',barmode='group')
df2.to_excel(f'{output_path}/speech_count_per_token_count_per_year.xlsx')
df2['order'] = df2.token_range.apply(lambda x: convert_table[x])
df2 = df2.sort_values(by=['year','order'])
display(px.line(df2, x='year', y='count', color='token_range'))

df3 = df.groupby(['year','token_range']).count()['who'].rename('count').reset_index()
df3['normalized_count'] = df3['count'] / df3.groupby('year')['count'].transform('sum')
df3.to_excel(f'{output_path}/speech_normalized_count_per_token_range_per_year.xlsx')
df3['order'] = df3.token_range.apply(lambda x: convert_table[x])
df3 = df3.sort_values(by='order')
display(px.area(df3, x='year', y='normalized_count', color='token_range'))

# %%
MD('# BELOW THIS, WE HAVE TEMPORARY TESTS; IGNORE')

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
