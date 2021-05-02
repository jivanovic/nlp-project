import nltk
import pandas as pd
import numpy as np
from data_extrafeatures import ExtraFeaturesData

class ModelData(ExtraFeaturesData):

    def build_final_model(self, word2vec_provider, stopwords=nltk.corpus.stopwords.words("english")):
        whitelist = self.whitelist
        stopwords = list(filter(lambda sw: sw not in whitelist, stopwords))
        extra_columns = [col for col in self.processed_data.columns if col.startswith("number_of")]
        similarity_columns = ["bad_similarity", "good_similarity", "information_similarity"]
        label_column = []
        if not self.is_testing:
            label_column = ["label"]

        columns = label_column + ["original_id"] + extra_columns + similarity_columns + list(
            map(lambda i: "word2vec_{0}".format(i), range(0, word2vec_provider.dimensions))) + list(
            map(lambda w: w + "_bow", self.wordlist))
        labels = []
        rows = []
        for idx in self.processed_data.index:
            current_row = []

            if not self.is_testing:
                # add label
                current_label = self.processed_data.loc[idx, "class"]
                labels.append(current_label)
                current_row.append(current_label)

            current_row.append(self.processed_data.loc[idx, "id"])

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_data.loc[idx, col])

            # average similarities with words
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            for main_word in map(lambda w: w.split("_")[0], similarity_columns):
                current_similarities = [abs(sim) for sim in
                                        map(lambda word: word2vec_provider.get_similarity(main_word, word.lower()),
                                            tokens) if
                                        sim is not None]
                if len(current_similarities) <= 1:
                    current_row.append(0 if len(current_similarities) == 0 else current_similarities[0])
                    continue
                max_sim = max(current_similarities)
                min_sim = min(current_similarities)
                current_similarities = [((sim - min_sim) / (max_sim - min_sim)) for sim in
                                        current_similarities]  # normalize to <0;1>
                current_row.append(np.array(current_similarities).mean())

            # add word2vec vector
            tokens = self.processed_data.loc[idx, "tokenized_text"]
            current_word2vec = []
            for _, word in enumerate(tokens):
                vec = word2vec_provider.get_vector(word.lower())
                if vec is not None:
                    current_word2vec.append(vec)

            #print(current_word2vec);
            if(len(current_word2vec) > 0):
                averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))
                for i in range(len(averaged_word2vec)):
                    if(np.isnan(averaged_word2vec[i])):
                        averaged_word2vec[i] = 0.0;
                current_row += averaged_word2vec

            # add bag-of-words
            tokens = set(self.processed_data.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)
        return self.data_model, self.data_labels