from __future__ import division
from __future__ import division

import csv
import inspect
import re
from collections import Counter
from csv import reader
from re import findall, sub, search
from time import time

# from nltk.tree import Tree
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class Feature:
    def __init__(self, utterance):
        self.utterance = utterance
        self.featureHeaders = [
            'question_mark',  # check for presence of question mark
            'wh_question',  # check for presence of wh- question words
            'i_dont_know',  # check for presence of phrase 'i don't know'
            'no_words',  # check for presence of "No" words
            'yes_words',  # check for presence of "Yes" words
            'do_words',  # check for presence of tense of "do" - did, does
            'non_verbal',  # check for presence of non-verbal words, < action >
            # 'UH_count',             # check for presence of Interjection (UH) Parts of speech in the sentence
            # 'CC_count',             # check for presence of co-ordinating conjunction (CC)
            'thanking_words',  # check for presence of words expressing "Thanks"
            'apology_words',  # check for presence of words
            # 'sub_utterance_index',  # add sub-utterance index
            # 'utterance_index',      # add utterance index
            # 'utterance_count'       # add conversation length
            # 'qrr_sequence'          # check for presence of speech tag "q<x>" in previous utterance and current occur
        ]

        self.featureKeys = {
            "question_mark": '?',
            "wh_question": ['who', 'which', 'where', 'what', 'how'],
            "i_dont_know": ["i don't know"],
            "no_words": ["no", "nah"],
            "yes_words": ["yes", "yeah"],
            "do_words": ["do", "did", "does"],
            "non_verbal": '^<.*?>',
            "UH_count": '/UH',
            "CC_count": '/CC',
            "thanking_words": ['thank', 'thanks', 'thank you'],
            "apology_words": ['sorry', 'apology'],
            "qrr_sequence": ['qw', 'qh', 'qo', 'qr']
        }

    def qrr_sequence(self):
        if len(self.previousUtterance_act_tag) != 0 and (
                self.previousUtterance_act_tag in self.featureKeys[inspect.currentframe().f_code.co_name]):
            return 1
        return 0

    def question_mark(self):
        if self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance[1]:
            return 1
        return 0

    def wh_question(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def i_dont_know(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def no_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def yes_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def do_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def non_verbal(self):
        # search for string <abcde>,
        #  ^ -> start of sentence, non-greedy pattern <.*?>
        return len(findall(self.featureKeys[inspect.currentframe().f_code.co_name], self.utterance[1]))

    def UH_count(self):
        # maybe, check for length of text; if length less than 2 then return true? - Skepticism :-/
        if len(self.utterance.pos.split()) < 3 and \
                self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance.pos:
            return 1
        return 0

    def CC_count(self):
        if len(self.utterance.pos.split()) < 3 and \
                self.featureKeys[inspect.currentframe().f_code.co_name] in self.utterance.pos:
            return 1
        return 0

    def thanking_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def apology_words(self):
        tag_word_count = 0
        for tag_word in self.featureKeys[inspect.currentframe().f_code.co_name]:
            if findall('\\b' + tag_word + '\\b', self.utterance[1]):
                tag_word_count += 1
        return tag_word_count

    def sub_utterance_index(self):
        return self.utterance.subutterance_index

    def utterance_index(self):
        return self.utterance.utterance_index

    def utterance_count(self):
        return self.utterance.utterance_count


class Transcript:
    """
    Transcript instances are basically just containers for lists of
    utterances and transcript-level metadata, accessible via
    attributes.
    """

    def __init__(self, swda_filename):
        """
        Sets up all the attribute values:
        Arguments:

        swda_filename -- the filename for this transcript
        metadata -- if a string, then assumed to be the metadata
        fileame, and the metadata is created from that filename if a
        Metadata object, then used as the needed metadata directly.
        """
        self.swda_filename = swda_filename
        # If the supplied value is a filename:
        # if isinstance(metadata, str) or isinstance(metadata, unicode):
        #    self.metadata = Metadata(metadata)
        # else: # Where the supplied value is already a Metadata object.
        #    self.metadata = metadata
        # Get the file rows:
        rows = list(csv.reader(self.swda_filename))
        # Ge the header and remove it from the rows:
        self.header = rows[0]
        rows.pop(0)
        # Extract the conversation_no to get the meta-data. Use the
        # header for this in case the column ordering is ever changed:
        row0dict = dict(zip(self.header, rows[1]))
        self.conversation_no = int(row0dict['conversation_no'])
        # The ptd filename in the right format for the current OS:
        # self.ptd_basename =  os.sep.join(row0dict['ptb_basename'].split("/"))
        # The dictionary of metadata for this transcript:
        # transcript_metadata = self.metadata[self.conversation_no]
        # for key, val in transcript_metadata.iteritems():
        #    setattr(self, key, transcript_metadata[key])
        utteranceCount = len(rows)
        # Create the utterance list:
        self.utterances = map((lambda x: Utterance(x, utteranceCount)), rows)
        # Coder's Manual: ``We also removed any line with a "@" (since @ marked slash-units with bad segmentation).''
        self.utterances = filter((lambda x: not re.search(r"[@]", x.act_tag)), self.utterances)


######################################################################

class Utterance:
    """
    The central object of interest. The attributes correspond to the
    values of the class variable header:
    'swda_filename':       (str) The filename: directory/basename
    'ptb_basename':        (str) The Treebank filename: add ".pos" for POS and ".mrg" for trees
    'conversation_no':     (int) The conversation Id, to key into the metadata database.
    'transcript_index':    (int) The line number of this item in the transcript (counting only utt lines).
    'act_tag':             (list of str) The Dialog Act Tags (separated by ||| in the file).
    'caller':              (str) A, B, @A, @B, @@A, @@B
    'utterance_index':     (int) The encoded index of the utterance (the number in A.49, B.27, etc.)
    'subutterance_index':  (int) Utterances can be broken across line. This gives the internal position.
    'text':                (str) The text of the utterance
    'pos':                 (str) The POS tagged version of the utterance, from PtbBasename+.pos
    """

    header = [
        'swda_filename',  # (str) The filename: directory/basename
        'ptb_basename',  # (str) The Treebank filename: add ".pos" for POS and ".mrg" for trees
        'conversation_no',  # (int) The conversation Id, to key into the metadata database.
        'transcript_index',  # (int) The line number of this item in the transcript (counting only utt lines).
        'act_tag',  # (list of str) The Dialog Act Tags (separated by ||| in the file).
        'caller',  # (str) A, B, @A, @B, @@A, @@B
        'utterance_index',  # (int) The encoded index of the utterance (the number in A.49, B.27, etc.)
        'subutterance_index',  # (int) Utterances can be broken across line. This gives the internal position.
        'text',  # (str) The text of the utterance
        'pos',  # (str) The POS tagged version of the utterance, from PtbBasename+.pos
        'utterance_count',  # (int) The count of utterances in the script
        'tokens',  # (list) Tokenized utterance
    ]

    def __init__(self, row, utteranceCount):
        """
        Arguments:
        row (list) -- a row from one of the corpus CSV files
        transcript_metadata (dict) -- a Metadata value based on the current conversation_no
        """
        ##################################################
        # Utterance data:
        for i in range(len(Utterance.header)):
            att_name = Utterance.header[i]
            row_value = None
            if i < len(row):
                row_value = row[i].strip()
            # Special handling of non-string values.
            if att_name == 'utterance_count':
                row_value = utteranceCount
            elif att_name == 'tokens':
                row_value = word_tokenize(getattr(self, 'text'))
            elif att_name == 'act_tag':
                # I thought these conjoined tags were meant to be split.
                # The docs suggest that they are single tags, thought,
                # so skip this conditional and let it be treated as a str.
                # row_value = re.split(r"\s*[,;]\s*", row_value)
                # `` Transcription errors (typos, obvious mistranscriptions) are marked with a "*" after
                # the discourse tag.''
                # These are removed for this version.
                row_value = row_value.replace("*", "")
            elif att_name in ('conversation_no', 'transcript_index', 'utterance_index', 'subutterance_index'):
                row_value = int(row_value)
            elif att_name == 'text':
                row_value = row_value.lower()
            # Add the attribute.
            setattr(self, att_name, row_value)
        ##################################################
        # Caller data:
        # for key in ('caller_sex', 'caller_education', 'caller_birth_year', 'caller_dialect_area'):
        #    full_key = 'from_' + key
        #    if self.caller.endswith("B"):
        #        full_key = 'to_' + key
        #    setattr(self, key, transcript_metadata[full_key])

    def damsl_act_tag(self):
        """
        Seeks to duplicate the tag simplification described at the
        Coders' Manual: http://www.stanford.edu/~jurafsky/ws97/manual.august1.html
        """
        d_tags = []
        tags = re.split(r"\s*[,;]\s*", self.act_tag)
        for tag in tags:
            if tag in ('qy^d', 'qw^d', 'b^m'):
                pass
            elif tag == 'nn^e':
                tag = 'ng'
            elif tag == 'ny^e':
                tag = 'na'
            else:
                tag = re.sub(r'(.)\^.*', r'\1', tag)
                tag = re.sub(r'[\(\)@*]', '', tag)
                if tag in ('qr', 'qy'):
                    tag = 'qy'
                elif tag in ('fe', 'ba'):
                    tag = 'ba'
                elif tag in ('oo', 'co', 'cc'):
                    tag = 'oo_co_cc'
                elif tag in ('fx', 'sv'):
                    tag = 'sv'
                elif tag in ('aap', 'am'):
                    tag = 'aap_am'
                elif tag in ('arp', 'nd'):
                    tag = 'arp_nd'
                elif tag in ('fo', 'o', 'fw', '"', 'by', 'bc'):
                    tag = 'fo_o_fw_"_by_bc'
            d_tags.append(tag)
        # Dan J says (p.c.) that it makes sense to take the first;
        # there are only a handful of examples with 2 tags here.
        return d_tags[0]

    def tree_is_perfect_match(self):
        """
        Returns True if self.trees is a singleton that perfectly matches
        the words in the utterances (with certain simplifactions to each
        to accommodate different notation and information).
        """
        if len(self.trees) != 1:
            return False
        tree_lems = self.regularize_tree_lemmas()
        pos_lems = self.regularize_pos_lemmas()
        if pos_lems == tree_lems:
            return True
        else:
            return False

    def regularize_tree_lemmas(self):
        """
        Simplify the (word, pos) tags asssociated with the lemmas for
        this utterances trees, so that they can be compared with those
        of self.pos. The output is a list of (string, pos) pairs.
        """
        tree_lems = self.tree_lemmas()
        tree_lems = filter((lambda x: x[1] not in ('-NONE-', '-DFL-')), tree_lems)
        tree_lems = map((lambda x: (re.sub(r"-$", "", x[0]), x[1])), tree_lems)
        return tree_lems

    def regularize_pos_lemmas(self):
        """
        Simplify the (word, pos) tags asssociated with self.pos, so
        that they can be compared with those of the trees. The output
        is a list of (string, pos) pairs.
        """
        pos_lems = self.pos_lemmas()
        pos_lems = filter((lambda x: len(x) == 2), pos_lems)
        pos_lems = filter((lambda x: x), pos_lems)
        nontree_nodes = ('^PRP^BES', '^FW', '^MD', '^MD^RB', '^PRP^VBZ', '^WP$', '^NN^HVS',
                         'NN|VBG', '^DT^BES', '^MD^VB', '^DT^JJ', '^PRP^HVS', '^NN^POS',
                         '^WP^BES', '^NN^BES', 'NN|CD', '^WDT', '^VB^PRP')
        pos_lems = filter((lambda x: x[1] not in nontree_nodes), pos_lems)
        pos_lems = filter((lambda x: x[0] != "--"), pos_lems)
        pos_lems = map((lambda x: (re.sub(r"-$", "", x[0]), x[1])), pos_lems)
        return pos_lems

    def text_words(self, filter_disfluency=False):
        """
        Tokenized version of the utterance; filter_disfluency=True
        will remove the special utterance notation to make the results
        look more like printed text. The tokenization itself is just
        spitting on whitespace, with no other simplification. The
        return value is a list of str instances.
        """
        t = self.text
        if filter_disfluency:
            t = re.sub(r"([+/\}\[\]]|\{\w)", "", t)
        return re.split(r"\s+", t.strip())

    def pos_words(self, wn_lemmatize=False):
        """
        Return the words associated with self.pos. wn_lemmatize=True
        runs the WordNet stemmer on the words before removing their
        tags.
        """
        lemmas = self.pos_lemmas(wn_lemmatize=wn_lemmatize)
        return [x[0] for x in lemmas]

    def tree_words(self, wn_lemmatize=False):
        """
        Return the words associated with self.trees
        terminals. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags.
        """
        lemmas = self.tree_lemmas(wn_lemmatize=wn_lemmatize)
        return [x[0] for x in lemmas]

    def pos_lemmas(self, wn_format=False, wn_lemmatize=False):
        """
        Return the (string, pos) pairs associated with
        self.pos. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags. wn_format merely changes the
        tags to wn_format where possible.
        """
        pos = self.pos
        pos = pos.strip()
        word_tag = map((lambda x: tuple(x.split("/"))), re.split(r"\s+", pos))
        word_tag = filter((lambda x: len(x) == 2), word_tag)
        word_tag = self.wn_lemmatizer(word_tag, wn_format=wn_format, wn_lemmatize=wn_lemmatize)
        return word_tag

    def tree_lemmas(self, wn_format=False, wn_lemmatize=False):
        """
        Return the (string, pos) pairs associated with self.trees
        terminals. wn_lemmatize=True runs the WordNet stemmer on the
        words before removing their tags. wn_format merely changes the
        tags to wn_format where possible.
        """
        word_tag = []
        for tree in self.trees:
            word_tag += tree.pos()
        return self.wn_lemmatizer(word_tag, wn_format=wn_format, wn_lemmatize=wn_lemmatize)

    def wn_lemmatizer(self, word_tag, wn_format=False, wn_lemmatize=False):
        # Lemmatizing implies converting to WordNet tags.
        if wn_lemmatize:
            word_tag = map(self.__treebank2wn_pos, word_tag)
            word_tag = map(self.__wn_lemmatize, word_tag)
        # This is tag conversion without lemmatizing.
        elif wn_format:
            word_tag = map(self.__treebank2wn_pos, word_tag)
        return word_tag

    def __treebank2wn_pos(self, lemma):
        """
        Internal method for turning a lemma's pos value into one that
        is compatible with WordNet, where possible (else the tag is
        left alone).
        """
        string, tag = lemma
        tag = tag.lower()
        if tag.startswith('v'):
            tag = 'v'
        elif tag.startswith('n'):
            tag = 'n'
        elif tag.startswith('j'):
            tag = 'a'
        elif tag.startswith('rb'):
            tag = 'r'
        return (string, tag)

    def __wn_lemmatize(self, lemma):
        """
        Lemmatize lemma using wordnet.stemWordNetLemmatizer(). Always
        returns a (string, pos) pair.  Lemmatizes even when the tag
        isn't helpful, by ignoring it for stemming.
        """
        string, tag = lemma
        wnl = WordNetLemmatizer()
        if tag in ('a', 'n', 'r', 'v'):
            string = wnl.lemmatize(string, tag)
        else:
            string = wnl.lemmatize(string)
        return (string, tag)


class Classifier:
    def __init__(self, dataset, dataset_train_path, dataset_test_path):
        self.dataName = dataset
        self.datasetTrainPath = dataset_train_path
        self.datasetTestPath = dataset_test_path
        self.speech_acts_class_count = Counter()
        self.data = []
        self.header = ['swda_filename', 'ptb_basename', 'conversation_no', 'transcript_index', 'act_tag', 'caller',
                       'utterance_index'
                       'subutterance_index', 'text', 'pos', 'trees', 'ptb_treenumbers']
        self.totalDataCount = 0
        self.trainData = []
        self.testData = []
        self.trainPercentage = 3
        self.testPercentage = 20
        self.speech_acts_class = [
            # 'sd',
            # 'b',
            # 'sv',
            # 'aa',
            'qy',
            # 'x',
            # 'ny',
            'qw',
            # 'nn',
            # 'h',
            # 'qy^d',
            # 'qw^d',
            'fa',
            'ft',
            'qrr',
            'qo',
            'qr',
            'ny',
            'nn',
            's'
        ]
        self.speech_acts_class = self.speechActDictify()

    def speechActDictify(self):
        speech_acts_class = Counter()
        for speech_act in self.speech_acts_class:
            speech_acts_class[speech_act] = 1

        return speech_acts_class

    def filterSpeechAct(self, speech_act):
        # speechActMod = sub('\^[a-z,0-9]+','', speechAct)
        found_S_tag = False
        speechActMod = speech_act.split('|')
        for act in speechActMod:
            if act == 's':
                found_S_tag = True
            elif self.speech_acts_class[act] != 0:
                return act

        speechActMod = speech_act.split('^')
        for act in speechActMod:
            if act == 's':
                found_S_tag = True
            elif self.speech_acts_class[act] != 0:
                return act

        speechActNorm = self.normalizemrdatosw(speech_act)
        if len(speechActNorm) != 0 and self.speech_acts_class[speechActNorm] != 0:
            speech_act = speechActNorm
        else:
            if found_S_tag:
                speech_act = 's'
            else:
                speech_act = "rest"

        return speech_act

    def getData(self, data_set_path, data):
        utterances = list(reader(data_set_path))
        utterances.pop(0)
        utteranceCount = len(utterances)
        for utter in utterances:
            utterInstance = []
            if utter[0].startswith('sw'):
                # Create the utterance list:
                # print utter
                utterance = map((lambda x: Utterance(x, utteranceCount)), [utter])
                # Coder's Manual: ``We also removed any line with a "@" (since @ marked slash-units with bad segmentation).''
                utterance = filter((lambda x: not search(r"[@]", x.act_tag)), utterance)
                # print utter, utterance
                if len(utterance) == 0:
                    continue
                utterance = utterance[0]
                utterID = utterance.swda_filename + str(utterance.utterance_index) + str(utterance.subutterance_index)
                utterText = utterance.text
                utterSpeechAct = utterance.act_tag
                utterTokens = utterance.tokens
            else:
                # utter = utter.split(',')
                utterID = utter[0]
                utterText = utter[1]
                utterSpeechAct = utter[3]
                utterTokens = word_tokenize(utterText)
                utterSpeechAct = self.filterSpeechAct(utterSpeechAct)

            self.speech_acts_class_count[utterSpeechAct] += 1
            self.totalDataCount += 1
            utterInstance.extend([utterID, utterText, utterSpeechAct, utterTokens])
            data.append(utterInstance)

    def getTrainAndTestData(self):
        self.trainData = self.data[:int(self.trainPercentage / 100 * self.totalDataCount)]
        self.testData = self.data[-int(self.testPercentage / 100 * self.totalDataCount):]

    def normalizemrdatosw(self, act_tag):
        tag = ''
        tmp = act_tag.split('|')
        tmp1 = []
        for j in range(len(tmp)):
            tmp1.extend(tmp[j].split('^'))
        if 's' in tmp1 and 'ar' in tmp1:
            tag = 'nn'
        elif 's' in tmp1 and 'aa' in tmp1:
            tag = 'ny'
        elif 's' in tmp1 and 'fa' in tmp1:
            tag = 'fa'
        elif 's' in tmp1 and 'ft' in tmp1:
            tag = 'ft'
        elif 's' in tmp1:
            tag = 's'

        return tag

    def combineFeatureVectors(self, feature_vectors_bow, feature_vectors_cust):
        feature_vectors = []
        for i in range(len(feature_vectors_bow)):
            feature_vectors.append(feature_vectors_bow[i] + feature_vectors_cust[i])
        return feature_vectors

    def featurize(self, utterances):
        feature_vectors = []
        speec_acts = []
        utter_text = []
        # form feature vector for sentences
        for utter in utterances:
            feature = Feature(utter)
            # feature_vector = {}
            feature_vector_utter = []
            for headers in feature.featureHeaders:
                # feature_vector[headers] = getattr(feature, headers)()
                feature_vector_utter.append(getattr(feature, headers)())
            speec_acts.append(utter[2])
            utter_text.append(utter[1])
            feature_vectors.append(feature_vector_utter)
            # previousUtter = utter[2]
            # feature_vectors.append([feature_vector[key] for key in feature_vector])
            # print utter[1], feature_vector

        return feature_vectors, speec_acts, utter_text

    def normalizeSpeechAct(self, speechActs):
        # normalize speech_acts
        for speechActIndex in range(len(speechActs)):
            trimSpeechAct = sub('\^2|\^g|\^m|\^r|\^e|\^q|\^d', '', speechActs[speechActIndex])
            if self.speech_acts_class[speechActs[speechActIndex]] != 1 or \
                    trimSpeechAct in ['sd', 'sv', 's'] or \
                    self.speech_acts_class[trimSpeechAct] != 1:
                # speechActs[speechActIndex] = 'other'
                speechActs[speechActIndex] = 'rest'

    def normalizeSpeechActTest(self, speechActs):
        # normalize speech_acts
        for speechActIndex in range(len(speechActs)):
            trimSpeechAct = sub('\^2|\^g|\^m|\^r|\^e|\^q|\^d', '', speechActs[speechActIndex])
            if trimSpeechAct in ['sd', 'sv']:
                speechActs[speechActIndex] = 's'
            elif self.speech_acts_class[speechActs[speechActIndex]] != 1 or \
                    self.speech_acts_class[trimSpeechAct] != 1:
                speechActs[speechActIndex] = 'rest'

    def normalizePrediction(self, predicted_speech_act, labelledSpeechAct):
        for i in range(len(labelledSpeechAct)):
            if labelledSpeechAct[i] == 's' and predicted_speech_act[i] == 'rest':
                predicted_speech_act[i] = 's'

    def combineFeatureVectors(self, feature_vectors_bow, feature_vectors_cust):
        feature_vectors = []
        for i in range(len(feature_vectors_bow)):
            feature_vectors.append(feature_vectors_bow[i] + feature_vectors_cust[i])
        return feature_vectors

    def findmajorityclass(self, speech_act):
        class_dist = Counter(speech_act)
        majority_class = class_dist.most_common(1)
        print("Majority class", majority_class)
        count = majority_class[0]
        print("Majority percentage: ", 100 * count[1] / len(speech_act))


classifier = Classifier('swa', '../Data/swda/')
dataStartTime = time()
classifier.getData()
dataEndTime = time()
print("Data loaded in", dataEndTime - dataStartTime, "sec")
# print(classifier.data[2].utterance_count)
# get test and train data
classifier.getTrainAndTestData()
featureStartTime = time()
# transform a feature vector
feature_vectors, speech_acts, utter_text = classifier.featurize(classifier.trainData)
featureEndTime = time()
print("Feature extracted in", featureEndTime - featureStartTime, "sec")
print(len(feature_vectors))
# normalize speech acts into classes
classifier.normalizeSpeechAct(speech_acts)
# train
trainStartTime = time()
clf = OneVsRestClassifier(SVC(C=1, kernel='poly', gamma='auto', verbose=False, probability=False))
clf.fit(feature_vectors, speech_acts)
trainEndTime = time()
print("Model trained in", trainEndTime - trainStartTime, "sec")
feature_vectors, labelled_speech_acts, utter_text = classifier.featurize(classifier.testData)
# normalize speech act for test data
classifier.normalizeSpeechAct(labelled_speech_acts)
# predict speech act for test
predicted_speech_act = clf.predict(feature_vectors)
correctResult = Counter()
wrongResult = Counter()
for i in range(len(predicted_speech_act)):
    if predicted_speech_act[i] == labelled_speech_acts[i]:
        correctResult[predicted_speech_act[i]] += 1
    else:
        wrongResult[predicted_speech_act[i]] += 1
total_correct = sum([correctResult[i] for i in correctResult])
total_wrong = len(predicted_speech_act) - total_correct
print("total_correct", total_correct)
print("total wrong", total_wrong)
print("accuracy", (total_correct / len(predicted_speech_act)) * 100)
print("Classification_report:\n", classification_report(labelled_speech_acts, predicted_speech_act))
# , target_names=target_names)
print("accuracy_score:", round(accuracy_score(labelled_speech_acts, predicted_speech_act), 2))
