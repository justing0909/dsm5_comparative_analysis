"""
DS 3500 Final Project
Analise Bottinger, Joshua Yu, Tyler Nguyen, Ryan Costa, Justin Guthrie

Creates the Sankey visualization for k most common words in DSM texts
"""

# Import statements
from collections import Counter, defaultdict
import pandas as pd
import sys
from extract_dsm import extract_text
import sankey as sk


def wordcount_counter(text, data, source_label):
    """

    :param text: list of each word in text
    :param data: dictionary of word counts for each DSM
    :param source_label: Signifies which DSM edition the words are being counted for
    :return:
    """
    if data is None:
        data = defaultdict(dict)
        data[source_label]['Text'] = text

    # Create a Word Counter
    word_counter = Counter(text)
    data[source_label]['Word Counter'] = word_counter
    return data


def wordcount_sankey(data, k=10):
    """
    Generate a sankey diagram from the texts to the set union of the k most common words of each text
    :param data: the text data in the DSMs
    :param k: most common words across the texts
    :return: Sankey Diagram in a separate window
    """

    assert type(k) == int, 'Expected int'
    assert k > 0, 'Expected positive k'

    try:
        # Store all dataframes for concatenation
        all_dfs = []

        # Loop through each DSM version and obtain the Word Counter for each book to store in a dataframe
        for text in data.keys():

            # Convert counter object stored in self.data to a dataframe
            temp_df = pd.DataFrame.from_dict(dict(data[text]['Word Counter'].most_common(k)),
                                             orient='index').reset_index()

            # Insert a DSM column to act as the src layer
            temp_df.insert(0, 'DSM Version', text)
            all_dfs.append(temp_df)

        # pd.concat will assume the set union of the k most common words
        df = pd.concat(all_dfs, axis=0)
        df = df.rename(columns={'DSM Version': 'DSM Version', 'index': 'Word', 0: 'Occurrence'})

        # src: DSM Version, targ: Word, val: Occurrence
        sk.make_sankey(df, 'DSM Version', 'Word', 'Occurrence')

    except AssertionError as ae:
        print("AssertionError:", ae, ": Line {}".format(sys.exc_info()[-1].tb_lineno))


def main():
    # Open DSM PDF files
    dsms = {'dsm1': 'dsm1.pdf',
            'dsm2': 'dsm2.pdf',
            'dsm3': 'dsm3.pdf',
            'dsm4': 'dsm4.pdf',
            'dsm5': 'dsm5.pdf'}

    # define start and end page numbers for each DSM version
    page_ranges = {'dsm1': (6, 141),
                   'dsm2': (8, 136),
                   'dsm3': (14, 507),
                   'dsm4': (16, 915),
                   'dsm5': (14, 992)}

    # load pdf files into library
    text_data = {}
    data = defaultdict(dict)
    for dsm, pdf_path in dsms.items():
        text_data[dsm] = extract_text(pdf_path, page_ranges[dsm])
        data = wordcount_counter(text_data[dsm].split(" "), data, source_label=dsm)

    # visualize using sankey
    wordcount_sankey(data)


if __name__ == "__main__":
    main()
