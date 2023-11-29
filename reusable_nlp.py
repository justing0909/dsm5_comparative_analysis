"""
Tyler Nguyen
DS3500
Reusable NLP / hw3
2/25/2023 / 3/1/2023
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import string
import plotly.graph_objects as go
import pandas as pd
from sankey import _code_mapping
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns


class nlp:

    def __init__(self):
        # manage data about the different texts that
        # we register with the framework
        self.data = defaultdict(dict)

    @staticmethod
    def _default_parser(text):
        """
        static method that parses text files
        :param: filename, a text file
        :return: return dictionary of results
        """

        # Open the file, read and clean punctuation, make all lowercase

        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )
        text = text.lower()

        # Take string and make words in individual tokens, load stopwords into stop variable, clean text from stopwords

        tokens = word_tokenize(text)
        stops = nlp._load_stop_words('stopfile.txt')
        text = [k for k in tokens if not k in stops]

        # Produce results in a dictionary

        results = {
            'word_count' : dict(Counter(text)),
            'base_text' : ' '.join(text),
            'sentiment' : TextBlob(' '.join(text)).sentiment
        }

        return results


    def _save_results(self, label, results):
        """
        Integrate parsing results into internal state
        label: unique label for a text file that we parsed
        results: the data extracted from the file as a dictionary attribute-->raw data
        """

        # save results into internal state

        for k, v in results.items():
            self.data[k][label] = v


    @staticmethod
    def _load_stop_words(stopfile):
        """
        static method that loads stop words and returns string of stopwords
        :param: stopfile, a list of stopwords
        :return: a string of stopwords
        """

        # read textfile and load stopwords into list

        f = open(stopfile)
        txt = [x.strip() for x in f.readlines()]
        return txt


    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework """

        # do default parsing of standard .txt file

        if parser is None:
            results = nlp._default_parser(filename)
        else:
            results = parser(filename)

        if label is None:
            label = filename

        # Save / integrate the data we extracted from the file
        # into the internal state of the framework

        self._save_results(label, results)

    def wordcount_sankey(self, word_list=None, k=5):
        """
        Creates a sankey diagram from internal state, allows number of words wanted on sankey diagram per instance
        :param: k, the number of highest frequency occuring words intended to be displayed
        :return: a sankey diagram
        """

        # open wordcount inside in saved data, acquire list of keys of wordcount

        wordcount = self.data['word_count']
        names = list(wordcount.keys())

        # Initiate three empty lists for sources, targs, and frequency to be used later in sankey

        srcs = []
        targs = []
        frequency = []

        # For loop that appends tuple of individual word, and the frequency of said word into temp list

        for name in names:
            templist = []
            for word in wordcount[name]:
                templist.append((word, wordcount[name][word]))

            # sort the templist by the number of occurrences

            templist.sort(key=lambda z:z[1], reverse=True)
            templist = templist[:k]

            # Initiate for loop inside temp list and append to srcs, targs, and frequency list as listed prior

            for x in templist:
                frequency.append(x[1])
                targs.append(x[0])
                srcs.append(name)

        # Initiate new dataframe, create columns from recently filled lists

        df = pd.DataFrame()
        df['srcs'] = srcs
        df['targs'] = targs
        df['frequency'] = frequency

        # use _code_mapping from sankey.py to map srcs onto targets numerically + map labels

        df, labels = _code_mapping(df, 'srcs', 'targs')

        # create link and node for sankey diagram

        link = {'source': df['srcs'], 'target': df['targs'], 'value': df['frequency']}
        node = {'label': labels}

        # initiate sankey and display graph

        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        fig.show()

    @staticmethod
    def varName(p):
        ''' Function:varname
            parameters: p, a variable
            Returns: the variable name as a string
        '''
        for k, v in globals().items():
            if id(p) == id(v):
                return k

    def make_wordclouds(self, title):
        """
        Make wordclouds in subplots from internal state
        :param: none
        :return: returns a subplot of wordclouds
        """

        # creates library base_text from base_text data stored in internal state

        base_text = self.data['base_text']

        # setting parameters for subplots

        fig = plt.figure(figsize = (5, 2))
        plt.title(title)
        plt.axis('off')


        # start counter to change position of subplots

        count = 1

        # for loop that produces graphs and locates position of each plot (some functions aren't working here!)

        for x in base_text:
            fig.add_subplot(5, 2, count)
            plt.title(nlp.varName(x))
            wc = WordCloud(width = 1000, height = 500).generate(base_text[x])
            plt.imshow(wc)
            plt.axis('off')
            count += 1

        # display plot

        plt.show()

    def make_sent_analysis(self):
        """
        Creates KDE plot and Scatter plot showing sentiment analysis of chosen libraries
        :param: none
        :return: returns a KDE plot and Scatter plot
        """

        # pull sentiment from internal state data, pull each value from tuple to acquire polarity and subjectivity

        sentiment = self.data['sentiment'].values()
        print(sentiment)
        polarity = [x[0] for x in sentiment]
        subjectivity = [x[1] for x in sentiment]

        # create plot with sentiment and polarity plotted against one another

        fig = plt.figure(figsize=(10, 10), dpi=100)
        plt.xlabel('Subjectivity')
        plt.ylabel('Polarity')
        plt.title('Sentiment analysis')
        sns.scatterplot(x=subjectivity, y=polarity, s=10)
        sns.kdeplot(x=subjectivity, y=polarity, color='black')
        plt.show()