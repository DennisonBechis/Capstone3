import pandas as pd
import numpy as np
import scattertext as st
import spacy
from pprint import pprint

def get_candidates(df):

    candidates_list = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar', 'Tom Steyer', 'Andrew Yang',
                        'Elizabeth Warren', 'Pete Buttigieg','Tulsi Gabbard', 'Kamala Harris', 'Cory Booker',
                        'Beto ORourke', 'Julian Castro', 'John Delaney', 'Kirsten Gillibrand', 'Bill de Blasio',
                        'Tim Ryan', 'Michael Bennet', "Beto O\x92Rourke",'Michael Bloomberg']

    return df[df['speaker'].isin(candidates_list)]

def assign_dropped(offense):

    still_running = ['Joe Biden','Bernie Sanders','Tom Steyer','Amy Klobuchar',
                        'Pete Buttigieg','Elizabeth Warren','Tulsi Gabbard']

    if offense in still_running:
        return 'running'
    else:
        return 'dropped'

def get_moderators(df):

    moderator_list = ['Speaker 1', 'Anderson Cooper', 'Jake Tapper', 'Chuck Todd', 'Rachel Maddow','George S.'
                        'Dana Bash', 'Erin Burnett', 'Lester Holt', 'Don Lemon', 'Judy Woodruff', 'David Muir',
                        'Marc Lacey', 'Marianne Williamson', 'Tim Alberta']

    return df[df['speaker'].isin(moderator_list)]

def unusable_rows(df):

    return df[df['speaking_time_seconds'] > 7]

if __name__ == "__main__":
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df = unusable_rows(df)
    df = get_candidates(df)
    df['still_running'] = df.apply(lambda row: assign_dropped(row[2]), axis=1)

    candidates_speeches = df.groupby('speaker').count().reset_index()
    candidates_speeches = candidates_speeches.sort_values(by='speech', ascending=False)

    nlp = st.WhitespaceNLP.whitespace_nlp
    corpus = st.CorpusFromPandas(df,
                         category_col='speaker',
                         text_col='speech',
                         nlp=nlp).build()

    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['Democratic Score'] = corpus.get_scaled_f_scores('Elizabeth Warren')
    pprint(list(term_freq_df.sort_values(by='Democratic Score', ascending=False).index[:20]))

    # html = st.produce_scattertext_explorer(corpus,
    #      category='running',
    #      category_name='running',
    #      not_category_name='dropped',
    #      width_in_pixels=1000,
    #      metadata=df['speaker'])
    # open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))
