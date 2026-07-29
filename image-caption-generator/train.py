import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Add, Dense, Dropout, Embedding, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import load_img, to_categorical


def image_features(images_dir, filenames):
    encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    features = {}
    for filename in sorted(set(filenames)):
        path = Path(images_dir) / filename
        if not path.exists():
            continue
        image = load_img(path, target_size=(299, 299))
        array = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(image), 0))
        features[filename] = encoder.predict(array, verbose=0)[0]
    return features


def build_model(vocab_size, max_length):
    image_input = Input(shape=(2048,))
    image_branch = Dropout(0.4)(image_input)
    image_branch = Dense(256, activation="relu")(image_branch)

    text_input = Input(shape=(max_length,))
    text_branch = Embedding(vocab_size, 256, mask_zero=True)(text_input)
    text_branch = Dropout(0.4)(text_branch)
    text_branch = LSTM(256)(text_branch)

    merged = Add()([image_branch, text_branch])
    output = Dense(vocab_size, activation="softmax")(merged)
    model = Model([image_input, text_input], output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def main(args):
    captions = pd.read_csv(args.captions).dropna(subset=["image", "caption"])
    captions["caption"] = "startseq " + captions["caption"].str.lower().str.strip() + " endseq"
    features = image_features(args.images, captions["image"])
    captions = captions[captions["image"].isin(features)]
    if captions.empty:
        raise ValueError("No caption rows matched image files.")

    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(captions["caption"])
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(text.split()) for text in captions["caption"])
    model = build_model(vocab_size, max_length)

    x_images, x_text, y = [], [], []
    for row in captions.itertuples(index=False):
        sequence = tokenizer.texts_to_sequences([row.caption])[0]
        for index in range(1, len(sequence)):
            x_images.append(features[row.image])
            x_text.append(pad_sequences([sequence[:index]], maxlen=max_length)[0])
            y.append(to_categorical(sequence[index], vocab_size))

    model.fit([np.array(x_images), np.array(x_text)], np.array(y), epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    model.save(artifacts / "caption_model.keras")
    (artifacts / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (artifacts / "config.json").write_text(json.dumps({"max_length": max_length}), encoding="utf-8")
    print("Saved model to artifacts/caption_model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    main(parser.parse_args())
