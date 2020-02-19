import pandas as pd
import numpy as np
import scattertext as st
import spacy
from pprint import pprint

df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')

def get_candidates(df):

    candidates_list = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar', 'Tom Steyer', 'Andrew Yang',
                        'Elizabeth Warren', 'Pete Buttigieg','Tulsi Gabbard', 'Kamala Harris', 'Cory Booker',
                        'Beto ORourke', 'Julian Castro', 'John Delaney', 'Kirsten Gillibrand', 'Bill de Blasio',
                        'Tim Ryan', 'Michael Bennet']

    return df[df['speaker'].isin(candidates_list)]

def get_moderators(df):

    moderator_list = ['Speaker 1', 'Anderson Cooper', 'Jake Tapper', 'Chuck Todd', 'Rachel Maddow','George S.'
                        'Dana Bash', 'Erin Burnett', 'Lester Holt', 'Don Lemon', 'Judy Woodruff', 'David Muir',
                        'Marc Lacey', 'Marianne Williamson', 'Tim Alberta']

    return df[df['speaker'].isin(moderator_list)]

def unusable_rows(df):

    return df[df['speaking_time_seconds'] > 4]

if __name__ == "__main__":
    df = unusable_rows(df)
    df = get_candidates(df)

    candidates_speeches = df.groupby('speaker').count().reset_index()
    candidates_speeches = candidates_speeches.sort_values(by='speech', ascending=False)
    print(candidates_speeches[['speaker','speech']])

    nlp = st.WhitespaceNLP.whitespace_nlp
    corpus = st.CorpusFromPandas(df,
                         category_col='speaker',
                         text_col='speech',
                         nlp=nlp).build()
    print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))
    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['Democratic Score'] = corpus.get_scaled_f_scores('Beto ORourke')
    pprint(list(term_freq_df.sort_values(by='Democratic Score', ascending=False).index[:10]))


    # print(a['speech'])
    # print(candidates_speeches.head(50))
    # print(candidates_speeches['speaker'].unique())
    # print(df['debate_name'].unique())
    # print(df[df['speaker'].isin(candidates_list)].info())
    # print(df[df['speaker'].isin(moderator_list)].info())
