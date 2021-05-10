import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, TFDistilBertForSequenceClassification
import matplotlib.pyplot as plt

data = pd.read_csv('data/data_mapbook.csv', quotechar="$", header=0, names=["id", "class", "text"], usecols=[1, 2])

# Transform positive/negative values to 1/0s
label_encoder = preprocessing.LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])
# data.head()
X = (np.array(data['text']))
y = (np.array(data['class']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
print("Train dataset shape: {0}, \nTest dataset shape: {1} \nValidation dataset shape: {2}".format(
    X_train.shape, X_test.shape, X_val.shape))

bert_model = TFDistilBertForSequenceClassification.from_pretrained("sampathkethineedi/industry-classification")
bert_tokenizer = BertTokenizer.from_pretrained("sampathkethineedi/industry-classification")

pad_token = 0
pad_token_segment_id = 0
max_length = 256


def plot_loss(history, label):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color='b', label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color='b', label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def convert_to_input(text):
    input_ids, attention_masks, token_type_ids = [], [], []

    for x in tqdm(text, position=0, leave=True):
        inputs = bert_tokenizer.encode_plus(x, add_special_tokens=True, max_length=max_length)

        i, t = inputs["input_ids"], inputs["token_type_ids"]
        m = [1] * len(i)

        padding_length = max_length - len(i)

        i = i + ([pad_token] * padding_length)
        m = m + ([0] * padding_length)
        t = t + ([pad_token_segment_id] * padding_length)

        input_ids.append(i)
        attention_masks.append(m)
        token_type_ids.append(t)

    return [np.asarray(input_ids), np.asarray(attention_masks), np.asarray(token_type_ids)]


X_test = convert_to_input(X_test)
X_train = convert_to_input(X_train)
X_val = convert_to_input(X_val)


def to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids, "attention_mask": attention_masks, "token_type_ids": token_type_ids}, y


train_ds = tf.data.Dataset.from_tensor_slices((X_train[0], X_train[1], X_train[2], y_train)).map(to_features) \
    .shuffle(100).batch(12).repeat(5)
val_ds = tf.data.Dataset.from_tensor_slices((X_val[0], X_val[1], X_val[2], y_val)).map(to_features).batch(12)
test_ds = tf.data.Dataset.from_tensor_slices((X_test[0], X_test[1], X_test[2], y_test)).map(to_features) \
    .batch(12)

# learning parameters
optimizer = tf.keras.optimizers.Adam(0.00000001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
bert_history = bert_model.fit(train_ds, epochs=100, validation_data=val_ds)
plot_loss(bert_history, 'loss')

# results
results_true = test_ds.unbatch()
results_true = np.asarray([element[1].numpy() for element in results_true])
print(results_true)

results = bert_model.predict(test_ds)
print(f"Model predictions:\n {results.logits}")

results_predicted = np.argmax(results.logits, axis=1)

print(f"F1 score: {f1_score(results_true, results_predicted, average='weighted')}")
print(f"Accuracy score: {accuracy_score(results_true, results_predicted)}")

# save model
bert_model.save_pretrained('models/bert_pretrained.h5')
