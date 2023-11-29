"""
Performs topic modelling and creates network graph for specific DSM section.
"""

# Import statements
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter
import pandas as pd


def fit_lda(preprocessed_text, num_topics, random_state=None):
    """
    Performs topic modelling using LDA on preprocessed text

    :param preprocessed_text: String of text to perform topic modelling on
    :param num_topics: Int number of topics
    :return topics: List of dictionaries per topic for a DSM edition, each topic contains 20 words
    """
    preprocessed_text = preprocessed_text.lower().split()
    # Convert the preprocessed text to a term-document matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preprocessed_text)

    # Fit the LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50, random_state=random_state)
    lda.fit(X)

    # Generate the list of topics with their top words
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-21:-1]]
        topics.append({'topic_idx': topic_idx, 'top_words': top_words, 'lda_component': topic})

    return topics


def plot_topics(topics_by_dsm, dsm_labels, top_n=20):
    """
    Plots topics by DSM edition for given DSM sections
    :param topics_by_dsm: List of topic dictionaries containing topics for each edition
    :param dsm_labels: Labels for topic groups
    :param top_n: Top n words to plot
    :return: Plots bar charts for words and their relative weights within a topic
    """
    num_dsms = len(topics_by_dsm)
    max_num_topics = max([len(topics) for topics in topics_by_dsm])

    subplot_titles = [f"{dsm_label} - Topic {i}" for dsm_label in dsm_labels for i in range(max_num_topics)]

    fig = make_subplots(rows=num_dsms, cols=max_num_topics, subplot_titles=subplot_titles)

    max_relative_weight = 0

    # Calculate the relative word weights for all topics
    for dsm_idx, (topics, dsm_label) in enumerate(zip(topics_by_dsm, dsm_labels)):
        for topic in topics:
            word_indices = [index for index, _ in sorted(enumerate(topic['lda_component']), key=lambda x: -x[1])][:top_n]
            word_weights = [topic['lda_component'][index] for index in word_indices]
            relative_word_weights = [weight / sum(topic['lda_component']) for weight in word_weights]
            topic['relative_word_weights'] = relative_word_weights
            max_relative_weight = max(max_relative_weight, max(relative_word_weights))

    for dsm_idx, (topics, dsm_label) in enumerate(zip(topics_by_dsm, dsm_labels)):
        for topic_idx, topic in enumerate(topics):
            top_words = topic['top_words'][:top_n]
            relative_word_weights = topic['relative_word_weights']

            trace = go.Bar(
                x=top_words,
                y=relative_word_weights,
                name=f"Topic {topic_idx}",
                marker=dict(color=f"rgba({30 * (topic_idx + 1)}, {30 * (dsm_idx + 1)}, 200, 0.7)"),
                showlegend=False,
                textposition='auto',
                textangle=-90
            )

            fig.add_trace(trace, row=dsm_idx + 1, col=topic_idx + 1)
            fig.update_yaxes(title_text="Relative Word Weight", row=dsm_idx + 1, col=topic_idx + 1, range=[0, max_relative_weight])
            fig.update_xaxes(tickangle=-90, tickfont=dict(size=10), row=dsm_idx + 1, col=topic_idx + 1)

    fig.update_layout(height=800 * num_dsms, width=350 * max_num_topics, title_text="DSM Topics",
                      title_x=0.5, template='plotly_white')

    fig.show()


def create_co_occurrence_matrix(text):
    """
    Creates co-occurence matrix for a given DSM section
    :param text: Pre-processed DSM section (string)
    :return matrix: Co-occurence matrix for top 100 words in text
    """
    # Count word occurrences and get the top 100 words
    words = text.lower().split()
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(100)]

    # Filter the text to only include the top 100 words
    filtered_text = [word for word in words if word in top_words]

    # Create the co-occurrence matrix for the filtered text
    matrix = pd.DataFrame(0, index=top_words, columns=top_words)
    for i, word in enumerate(filtered_text[:-1]):
        for j in range(1, min(6, len(filtered_text) - i)):
            next_word = filtered_text[i + j]
            matrix.at[word, next_word] += 1
            matrix.at[next_word, word] += 1

    return matrix


def create_network_graph(matrix, labels, title, threshold=0.00095):
    """
    Creates network graph based on co-occurrence matrix for DSM section
    :param matrix: Co-occurrence matrix (np array)
    :param labels: List of words to label each node in graph based on DSM section
    :param title: Title of graph
    :param threshold: Threshold to filter graph edges based on relative weights
    :return: Plots network graph of co-occurring words in DSM text
    """
    G = nx.from_numpy_array(matrix)

    # Calculate total weight of all edges
    total_weight = sum([d['weight'] for u, v, d in G.edges(data=True)])

    # Filter edges below the threshold proportion
    edges_to_remove = [(u, v) for u, v in G.edges() if (G[u][v]['weight'] / total_weight) <= threshold]
    G.remove_edges_from(edges_to_remove)

    # Remove nodes with a degree of less than or equal to 1
    nodes_to_remove = [node for node, degree in G.degree() if degree <= 2]
    G.remove_nodes_from(nodes_to_remove)

    # Create a layout for the nodes
    pos = nx.spring_layout(G, k=0.35)

    # Prepare the edge traces
    edge_traces = []
    max_weight = max([d['weight'] for u, v, d in G.edges(data=True)])
    cmap = plt.cm.viridis_r

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        # Normalize weights
        norm_weight = weight / max_weight
        edge_color = 'rgba' + str(cmap(norm_weight, bytes=True))[:-2] + ')'

        edge_trace = go.Scatter(x=[x0, x1], y=[y0, y1], line=dict(width=2, color=edge_color),
                                hoverinfo='none', mode='lines')
        edge_traces.append(edge_trace)

    # Create the node trace
    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers+text', textposition='top center',
                            hoverinfo='text', marker=dict(showscale=True, colorscale='Viridis',
                                                          reversescale=True, color=[], size=20,
                                                          colorbar=dict(thickness=15,
                                                                        title='Co-occurrence Frequency', xanchor='left',
                                                                        titleside='right', x=1.0, y=0.5),
                                                          line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([labels[node]])
        node_trace['marker']['color'] += tuple([sum(matrix[node])])

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(title=title, showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.1, 0.75]),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.show()
