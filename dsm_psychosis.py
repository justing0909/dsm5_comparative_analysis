"""
DS 3500 Final Project
Analise Bottinger, Joshua Yu, Tyler Nguyen, Ryan Costa, Justin Guthrie

Analyzes psychosis/schizophrenia section of DSM
"""

# Import statements
from extract_dsm import extract_text
import analyze_dsm_sect

# Global variables
DSM1 = "dsm1.pdf"
DSM2 = "dsm2.pdf"
DSM3 = "dsm3.pdf"
DSM4 = "dsm4.pdf"
DSM5 = "dsm5.pdf"


def main():
    # Load in dsm 3 depressive disorders section
    dsm3_psychosis = extract_text(DSM3, (194, 216))
    # Load in dsm 4 depressive disorders section
    dsm4_psychosis = extract_text(DSM4, (302, 344))
    # Load in dsm 5 depressive disorders section
    dsm5_psychosis = extract_text(DSM5, (132, 167))

    dsm3_topics = analyze_dsm_sect.fit_lda(dsm3_psychosis, 3, random_state=42)
    dsm4_topics = analyze_dsm_sect.fit_lda(dsm4_psychosis, 3, random_state=42)
    dsm5_topics = analyze_dsm_sect.fit_lda(dsm5_psychosis, 3, random_state=42)

    topics_by_dsm = [dsm3_topics, dsm4_topics, dsm5_topics]
    dsm_labels = ["DSM-3", "DSM-4", "DSM-5"]
    analyze_dsm_sect.plot_topics(topics_by_dsm, dsm_labels)

    # Calculate co-occurrence matrices
    dsm3_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm3_psychosis)
    dsm4_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm4_psychosis)
    dsm5_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm5_psychosis)

    # Get the unique words from the DSM texts
    words_dsm3 = list(dsm3_matrix.columns)
    words_dsm4 = list(dsm4_matrix.columns)
    words_dsm5 = list(dsm5_matrix.columns)

    # Create network graphs using the co-occurrence matrices
    analyze_dsm_sect.create_network_graph(dsm3_matrix.values, words_dsm3, "Co-occurrence Network Graph for DSM-3 Psychosis")
    analyze_dsm_sect.create_network_graph(dsm4_matrix.values, words_dsm4, "Co-occurrence Network Graph for DSM-4 Psychosis")
    analyze_dsm_sect.create_network_graph(dsm5_matrix.values, words_dsm5, "Co-occurrence Network Graph for DSM-5 Psychosis")

    # SANKEY, SENTIMENT ANALYSIS, WORDCLOUD
    from reusable_nlp import nlp

    dsm_psy = nlp()
    dsm_psy.load_text(filename=dsm3_psychosis, label='dsm3')
    dsm_psy.load_text(filename=dsm4_psychosis, label='dsm4')
    dsm_psy.load_text(filename=dsm5_psychosis, label='dsm5')

    dsm_psy.make_sent_analysis()
    dsm_psy.wordcount_sankey()


if __name__ == "__main__":
    main()