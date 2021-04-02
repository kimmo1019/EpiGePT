import tensorflow as tf 

_NEG_INF_FP32 = -1e9

def get_padding(x, padding_value=0, dtype=tf.float32):
    """Return float tensor representing the padding values in x.
    Args:
        x: int tensor with any shape
        padding_value: int which represents padded values in input
        dtype: The dtype of the return value.
    Returns:
        float tensor with same shape as x containing values 0 or 1.
            0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x, padding_value=0, dtype=tf.float32):
    """Calculate bias tensor from padding values in tensor.
    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.
    Args:
        x: int tensor with shape [batch_size, length]
        padding_value: int which represents padded values in input
        dtype: The dtype of the return value
    Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x, padding_value, dtype)
        attention_bias = padding * _NEG_INF_FP32
        attention_bias = tf.expand_dims(
                tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias