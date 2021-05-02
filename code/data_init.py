import pandas as pd

class InitializeData():
    data = []
    processed_data = []
    wordlist = []

    data_model = None
    data_labels = None
    is_testing = False

    def initialize(self, csv_file, is_testing_set=False, from_cached=None):
        if from_cached is not None:
            self.data_model = pd.read_csv(from_cached)
            return

        self.is_testing = is_testing_set

        if not is_testing_set:
            #self.data = pd.read_csv(csv_file, header=0, names=["id", "class", "text"])
            self.data = pd.read_csv(csv_file, quotechar="$", header=0, names=["id", "class", "text"])
            # "Assignment Instructions" "Emoticon/Non-verbal" "Response"
            self.data = self.data[self.data["class"].isin(["Content Discussion", "Greeting", "Logistics", "Instruction Question", "Assignment Question", "General Comment", "Incomplete/typo", "Feedback", "Discussion Wrap-up", "Outside Material", "Opening Statement", "General Question", "Content Question", "Emoticon/Non-verbal", "Assignment Instructions", "Response"])]
            #self.data = self.data[self.data["class"].isin(["positive", "negative", "neutral"])]
        else:
            self.data = pd.read_csv(csv_file, header=0, names=["id", "text"], dtype={"id": "int64", "text": "str"},
                                    nrows=4000)
            not_null_text = 1 ^ pd.isnull(self.data["text"])
            not_null_id = 1 ^ pd.isnull(self.data["id"])
            self.data = self.data.loc[not_null_id & not_null_text, :]

        self.processed_data = self.data
        self.wordlist = []
        self.data_model = None
        self.data_labels = None