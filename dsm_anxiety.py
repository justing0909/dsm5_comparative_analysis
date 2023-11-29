"""
DS 3500 Final Project
Analise Bottinger, Joshua Yu, Tyler Nguyen, Ryan Costa, Justin Guthrie

Analyzes anxiety section of DSM
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

    # Load in dsm 3 anxiety disorders section
    dsm3_anxiety = extract_text(DSM3, (234, 278))
    # Load in dsm 4 anxiety disorders section
    dsm4_anxiety = extract_text(DSM4, (422, 473))
    # Load in dsm 5 anxiety disorders section
    dsm5_anxiety = extract_text(DSM5, (238, 252))

    dsm3_topics = analyze_dsm_sect.fit_lda(dsm3_anxiety, 3, random_state=42)
    dsm4_topics = analyze_dsm_sect.fit_lda(dsm4_anxiety, 3, random_state=42)
    dsm5_topics = analyze_dsm_sect.fit_lda(dsm5_anxiety, 3, random_state=42)

    topics_by_dsm = [dsm3_topics, dsm4_topics, dsm5_topics]
    dsm_labels = ["DSM-3", "DSM-4", "DSM-5"]
    analyze_dsm_sect.plot_topics(topics_by_dsm, dsm_labels)

    # Calculate co-occurrence matrices
    dsm3_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm3_anxiety)
    dsm4_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm4_anxiety)
    dsm5_matrix = analyze_dsm_sect.create_co_occurrence_matrix(dsm5_anxiety)

    # Get the unique words from the DSM texts
    words_dsm3 = list(dsm3_matrix.columns)
    words_dsm4 = list(dsm4_matrix.columns)
    words_dsm5 = list(dsm5_matrix.columns)

    # Create network graphs using the co-occurrence matrices
    analyze_dsm_sect.create_network_graph(dsm3_matrix.values, words_dsm3, "Co-occurrence Network Graph for DSM-3 Anxiety")
    analyze_dsm_sect.create_network_graph(dsm4_matrix.values, words_dsm4, "Co-occurrence Network Graph for DSM-4 Anxiety")
    analyze_dsm_sect.create_network_graph(dsm5_matrix.values, words_dsm5, "Co-occurrence Network Graph for DSM-5 Anxiety")


if __name__ == "__main__":
    main()
