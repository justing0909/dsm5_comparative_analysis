"""
DS 3500 Final Project
Analise Bottinger, Joshua Yu, Tyler Nguyen, Ryan Costa, Justin Guthrie

Analyzes depression section of DSM
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
    dsm3_depression = extract_text(DSM3, (218, 237))
    # Load in dsm 4 depressive disorders section
    dsm4_depression = extract_text(DSM4, (346, 420))
    # Load in dsm 5 depressive disorders section
    dsm5_depression = extract_text(DSM5, (200, 233))

    dsm3_topics = analyze_dsm_sect.fit_lda(dsm3_depression, 3, random_state=42)
    dsm4_topics = analyze_dsm_sect.fit_lda(dsm4_depression, 3, random_state=42)
    dsm5_topics = analyze_dsm_sect.fit_lda(dsm5_depression, 3, random_state=42)

    topics_by_dsm = [dsm3_topics, dsm4_topics, dsm5_topics]
    dsm_labels = ["DSM-3", "DSM-4", "DSM-5"]
    analyze_dsm_sect.plot_topics(topics_by_dsm, dsm_labels)

    # Calculate co-occurrence matrices
    dsm3_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm3_depression)
    dsm4_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm4_depression)
    dsm5_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm5_depression)

    # Get the unique words from the DSM texts
    words_dsm3 = list(dsm3_matrix.columns)
    words_dsm4 = list(dsm4_matrix.columns)
    words_dsm5 = list(dsm5_matrix.columns)

    # Create network graphs using the co-occurrence matrices
    analyze_dsm_sect.create_network_graph(dsm3_matrix.values, words_dsm3, "Co-occurrence Network Graph for DSM-3 Depression")
    analyze_dsm_sect.create_network_graph(dsm4_matrix.values, words_dsm4, "Co-occurrence Network Graph for DSM-4 Depression")
    analyze_dsm_sect.create_network_graph(dsm5_matrix.values, words_dsm5, "Co-occurrence Network Graph for DSM-5 Depression")


if __name__ == "__main__":
    main()
