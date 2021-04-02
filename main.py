import tensorflow as tf
import numpy as np 
import sys, os 
from absl import app, flags, logging
from geformer import Geformer

DTYPE_MAP = {
    "fp16": tf.float16,
    "bf16": tf.bfloat16,
    "fp32": tf.float32,
}

def train(params):
    model = Geformer(params)
    optimizer = tf.keras.optimizers.Adam(learning_rate = FLAGS.lr)
    model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())
    seq_test = np.random.normal(size = (1000, 1000*2**params["num_cb"], 1, 4))
    expr_test = np.random.normal(size = (1000, 1000, params["hidden_size"]-params["num_channels"]))
    target_test = np.random.normal(size = (1000, 1000, 1000))
    model.fit([seq_test, expr_test], target_test, epochs=10, batch_size=FLAGS.bs)
    out = model.predict([seq_test, expr_test])
    print(out.shape)


def main(argv):
    del argv
    params = {}
    params["num_channels"] = FLAGS.num_channels
    params["num_cb"] = FLAGS.num_cb
    params["hidden_size"] = FLAGS.hidden_size
    params["num_hidden_layers"] = FLAGS.num_hidden_layers
    params["num_heads"] = FLAGS.num_heads
    params["attention_dropout"] = FLAGS.attention_dropout
    params["filter_size"] = FLAGS.filter_size
    params["relu_dropout"] = FLAGS.relu_dropout
    params["layer_postprocess_dropout"] = FLAGS.layer_postprocess_dropout
    params["dtype"] = DTYPE_MAP[FLAGS.dtype]
    print(params)
    train(params)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("num_channels", 300, "Output channels number for convolution stack")
    flags.DEFINE_integer("num_cb", 8, "Number of convolution blocks")
    flags.DEFINE_integer("hidden_size", 1000, "word embedding dimension")
    flags.DEFINE_integer("num_hidden_layers", 10, "number of MultiHeadAttention")
    flags.DEFINE_integer("num_heads", 8, "number of heads in each MHA")
    flags.DEFINE_float("attention_dropout", 0.1, "attention dropout rate")
    flags.DEFINE_integer("filter_size", 512, "filter size for the inner (first) dense layer in ffn")
    flags.DEFINE_float("relu_dropout", 0.1, "ffn dropout rate")
    flags.DEFINE_float("layer_postprocess_dropout", 0.1, "encoder inputs dropout rate")
    flags.DEFINE_float("lr", 1e-4, "learning rate")
    flags.DEFINE_integer("bs", 32, "batch size")
    flags.DEFINE_integer("epochs", 100, "epochs for training")
    flags.DEFINE_string("dtype", "fp32", "Data type for tensor")
    app.run(main) 
