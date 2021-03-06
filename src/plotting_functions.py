import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from debate_cleaning import *

def horizontal_bar(ax, labels, data, name='horizontal_bar_graph', x_label = False, y_label = False, title = False):

    colors = ['tab:blue','tab:orange','tab:red','tab:green','tab:purple','tab:brown',
                    'tab:pink','tab:cyan','tab:olive','tab:blue','tab:grey','tab:brown']
    length_labels = np.arange(len(labels))
    ax.barh(length_labels, data, color=colors)
    ax.set_ylabel(y_label, fontsize = 16)
    ax.set_xlabel(x_label, fontsize = 16)
    ax.set_yticks(length_labels)
    ax.set_xticks([0,50,100,150,200,250,300])
    ax.set_xticklabels([0,50,100,150,200,250,300], fontsize=16)
    ax.set_yticklabels(labels, fontsize = 16)
    ax.set_title(title, fontsize = 20)
    plt.tight_layout()
    plt.savefig('/Users/bechis/dsi/repo/Capstone3/images'+name+'.png', transparent = True)
    return ax

def line_plot(ax, x, y, label):
    ax.plot(x, y, label=label)
    return ax

def bar_plot(ax, X_labels, Y_axis, name= 'bar_plot', x_name = False, y_name = False, color = 'blue', title =''):

    X_ticks = [x for x in range(0,len(X_labels))]
    ax.bar(X_ticks, Y_axis, color= color, align= 'center', width=1)
    ax.set_xticks(ticks = X_ticks)
    ax.set_xticklabels(X_labels)
    ax.yaxis.grid(True)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig('../images/'+name+'.png', transparent = True)
    return ax

if __name__ == "__main__":
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df1 = pd.read_csv("/Users/bechis/DSI/repo/capstone3/data/debate_transcripts_nevada.csv", encoding= 'unicode_escape')
    df2 = pd.read_csv("/Users/bechis/dsi/repo/capstone3/data/South_Carolina_Debate.csv", encoding= 'unicode_escape')
    df = pd.concat([df, df1, df2])
    df = unusable_rows(df)
    df = get_candidates(df)

    grouped_candidates = df.groupby(['speaker']).count().reset_index()
    grouped_candidates = grouped_candidates.sort_values(by='debate_name', ascending=True)

    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(1,1,1)
    
    d = grouped_candidates['speaker'].to_numpy()
    d[8] = "Beto O'Rourke"

    horizontal_bar(ax, grouped_candidates['speaker'].to_numpy(), grouped_candidates['debate_name'],name = 'Debates_attended2', x_label = 'Times Spoken',
                     y_label= 'Democratic Candidates', title = 'Debates Attended')
