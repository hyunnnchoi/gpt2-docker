# 27/07/2024, EDITED BY HYUNMOK CHOI
import keras
import keras_nlp
import benchmark
from benchmark import utils
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.distribute import MultiWorkerMirroredStrategy

os.environ["KERAS_BACKEND"] = "tensorflow"

def get_model(preprocessor):
    model = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    model.backbone.enable_lora(rank=4)
    return model

def run(batch_size=benchmark.GPT2_FIT_BATCH_SIZE):
    strategy = MultiWorkerMirroredStrategy()
    with strategy.scope():
        if hasattr(keras, "config"):
            keras.config.set_dtype_policy(benchmark.FLOAT_T4)
        else:
            keras.mixed_precision.set_global_policy(benchmark.FLOAT_T4)
        
        preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
            "gpt2_base_en",
            sequence_length=benchmark.GPT2_SEQ_LENGTH,
        )
        model = get_model(preprocessor)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.AdamW(),
            jit_compile=utils.use_jit(),
        )

    reddit_ds = tfds.load("reddit", split="train", as_supervised=True)
    train_ds = (
        reddit_ds.map(lambda document, _: document)
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    NUM_BATCHES = 500
    dataset = train_ds.take(NUM_BATCHES)
    return utils.fit(model, dataset)

if __name__ == "__main__":
    benchmark.benchmark(run)
