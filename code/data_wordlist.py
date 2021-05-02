from collections import Counter
import nltk
import pandas as pd
from data_token_stem import TokenStemData

class WordlistData(TokenStemData):
    def __init__(self, previous):
        self.processed_data = previous.processed_data

    whitelist = ["n't", "not"]
    wordlist = []

    def build_wordlist(self, min_occurrences=3, max_occurences=500, stopwords=nltk.corpus.stopwords.words("english"),
                       whitelist=None):
        self.wordlist = []
        whitelist = self.whitelist if whitelist is None else whitelist
        import os
        if os.path.isfile("data/wordlist.csv"):
            word_df = pd.read_csv("data/wordlist.csv")
            word_df = word_df[word_df["occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "word"])
            return

        words = Counter()
        for idx in self.processed_data.index:
            words.update(self.processed_data.loc[idx, "text"])

        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]

        word_df = pd.DataFrame(
            data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                  "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
            columns=["word", "occurrences"])

        word_df.to_csv("data/wordlist.csv", index_label="idx")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]