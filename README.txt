In order to run this code, please have the following libraries installed:

random
networkx
numpy
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
matplotlib.pyplot

From there, open a desired IDE and run the main.py file.
This should work for any python in version 3.

Running the code will generate/replace 2 files:

scatter.png - the scatter plot generated from node2vec, with each color representing the
true division in football.

nfl.model - the node2vec model used for modeling links and similarities.

results - all findings, including statistical analysis such as precision, recall, and F1-score
outputted in the terminal
