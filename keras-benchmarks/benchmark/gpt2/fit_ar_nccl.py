import keras
import keras_nlp

import benchmark
from benchmark import utils

import os
import json

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import time

def get_model(preprocessor):
    #model = keras_nlp.models.MistralCausalLM.from_preset(
    #    "mistral_7b_en",
    #    preprocessor=preprocessor,
    #)
    model = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    model.backbone.enable_lora(rank=4)
    return model

def run(batch_size=benchmark.GPT2_FIT_BATCH_SIZE):
    options = tf.distribute.experimental.CommunicationOptions(
        bytes_per_pack=0,
        timeout_seconds=None,
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
    )
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=options)
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])
    global_batch_size = batch_size

    if hasattr(keras, "config"):
        keras.config.set_dtype_policy(benchmark.FLOAT_T4)
    else:
        keras.mixed_precision.set_global_policy(benchmark.FLOAT_T4)

    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=benchmark.GPT2_SEQ_LENGTH,
    )

    def dataset_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        reddit_ds = tfds.load("reddit", split="train", as_supervised=True, data_dir='/data/reddit')
        dataset = (
            reddit_ds.shard(
                num_shards=input_context.num_input_pipelines,
                index=input_context.input_pipeline_id
            )
            .map(lambda document, _: document)
            .batch(batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        NUM_BATCHES = 500
        return dataset.take(NUM_BATCHES)

    dataset = strategy.distribute_datasets_from_function(dataset_fn)

    with strategy.scope():
        model = get_model(preprocessor)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.AdamW(),
            jit_compile=utils.use_jit(),
        )
    return utils.fit(model, dataset)

if __name__ == "__main__":
    benchmark.benchmark(run)
