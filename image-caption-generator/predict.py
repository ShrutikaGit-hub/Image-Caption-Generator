import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


def main(image_path):
    artifacts = Path("artifacts")
    model = tf.keras.models.load_model(artifacts / "caption_model.keras")
    tokenizer = tokenizer_from_json((artifacts / "tokenizer.json").read_text(encoding="utf-8"))
    max_length = json.loads((artifacts / "config.json").read_text())["max_length"]
    encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    image = tf.keras.utils.load_img(image_path, target_size=(299, 299))
    image = preprocess_input(np.expand_dims(tf.keras.utils.img_to_array(image), 0))
    feature = encoder.predict(image, verbose=0)
    caption = "startseq"
    reverse_index = {value: key for key, value in tokenizer.word_index.items()}
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        prediction = model.predict([feature, pad_sequences([sequence], maxlen=max_length)], verbose=0)
        word = reverse_index.get(int(np.argmax(prediction)), "")
        if not word or word == "endseq":
            break
        caption += " " + word
    print(caption.replace("startseq ", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    main(parser.parse_args().image)
