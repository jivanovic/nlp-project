from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, TFBertMainLayer, \
    TFDistilBertForSequenceClassification, TFDistilBertPreTrainedModel, TFDistilBertMainLayer
from tensorflow.keras import layers

import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


class BertIMDBEmbeddingModel(TFDistilBertPreTrainedModel):
    def __init__(self, config,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model",
                 *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFDistilBertMainLayer(config, name="bert", trainable=True)

        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training=False, **kwargs):
        bert_outputs = self.bert(inputs, training=training, **kwargs)

        l_1 = self.cnn_layer1(bert_outputs[0])
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(bert_outputs[0])
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(bert_outputs[0])
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)  # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


data = pd.read_csv('data/data_mapbook.csv', quotechar="$", header=0, names=["id", "class", "text"], usecols=[1, 2])

label_encoder = preprocessing.LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])
# data.head()
X = (np.array(data['text']))
y = (np.array(data['class']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
print("Train dataset shape: {0}, \nTest dataset shape: {1} \nValidation dataset shape: {2}".format(
    X_train.shape, X_test.shape, X_val.shape))

# bert_model = TFDistilBertForSequenceClassification.from_pretrained("sampathkethineedi/industry-classification")
bert_tokenizer = BertTokenizer.from_pretrained("sampathkethineedi/industry-classification")


def get_token_ids(texts):
    return bert_tokenizer.batch_encode_plus(texts,
                                            add_special_tokens=True,
                                            max_length=128,
                                            pad_to_max_length=True)["input_ids"]


train_token_ids = get_token_ids(X_train)
test_token_ids = get_token_ids(X_test)
train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_token_ids), tf.constant(y_train))).batch(12)
test_data = tf.data.Dataset.from_tensor_slices((tf.constant(test_token_ids), tf.constant(y_test))).batch(12)

CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 16
DROPOUT_RATE = 0.2
NB_EPOCHS = 5

text_model = BertIMDBEmbeddingModel.from_pretrained('sampathkethineedi/industry-classification',
                                                    cnn_filters=CNN_FILTERS,
                                                    dnn_units=DNN_UNITS,
                                                    model_output_classes=OUTPUT_CLASSES,
                                                    dropout_rate=DROPOUT_RATE)


if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])

text_model.fit(train_data, epochs=NB_EPOCHS)

results_predicted = [text_model.predict(test_data) ]
results_true = np.array(y_test)
print(f"F1 score: {f1_score(results_true, results_predicted)}")
print(f"Accuracy score: {accuracy_score(results_true, results_predicted)}")