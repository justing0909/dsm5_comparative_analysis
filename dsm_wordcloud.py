"""
DS 3500 Final Project
Analise Bottinger, Joshua Yu, Tyler Nguyen, Ryan Costa, Justin Guthrie

Creates Word Clouds of 100 most common words in each DSM Edition
"""

# Import statements
import matplotlib.pyplot as plt
from extract_dsm import extract_text
from wordcloud import WordCloud, STOPWORDS
from collections import Counter


# Global variables
DSM1 = "dsm1.pdf"
DSM2 = "dsm2.pdf"
DSM3 = "dsm3.pdf"
DSM4 = "dsm4.pdf"
DSM5 = "dsm5.pdf"


def create_word_cloud(keywords, maximum_words=100, bg='white', cmap='Dark2',
                      maximum_font_size=256, width=3000, height=2000,
                      random_state=42, fig_w=15, fig_h=10, output_filepath=None):
    # Convert keywords to dictionary with values and its occurences
    word_cloud_dict = Counter(keywords)

    wordcloud = WordCloud(background_color=bg, max_words=maximum_words, colormap=cmap,
                          stopwords=STOPWORDS, max_font_size=maximum_font_size,
                          random_state=random_state,
                          width=width, height=height).generate_from_frequencies(word_cloud_dict)

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    dsm1 = extract_text(DSM1, (6, 141))
    dsm2 = extract_text(DSM2, (8, 136))
    dsm3 = extract_text(DSM3, (14, 507))
    dsm4 = extract_text(DSM4, (16, 915))
    dsm5 = extract_text(DSM5, (14, 992))

    create_word_cloud(dsm1.split(" "))
    create_word_cloud(dsm2.split(" "))
    create_word_cloud(dsm3.split(" "))
    create_word_cloud(dsm4.split(" "))
    create_word_cloud(dsm5.split(" "))

if __name__ == "__main__":
    main()
