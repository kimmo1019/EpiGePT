import tensorflow as tf
import utils
from layers import ffn_layer
from layers import attention_layer
from layers import prepost_layer
from layers import embedding_layer
from layers import position_embedding


class EncoderStack(tf.keras.layers.Layer):
      """Transformer encoder stack.
  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape):
    """Builds the encoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          prepost_layer.PrePostProcessingWrapper(self_attention_layer, params),
          prepost_layer.PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, encoder_inputs, attention_bias, inputs_padding, training):
    """Return the output of the encoder layer stacks.
    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.
    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          encoder_inputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
        with tf.name_scope("ffn"):
          encoder_inputs = feed_forward_network(
              encoder_inputs, training=training)

    return self.output_normalization(encoder_inputs)


class Geformer(tf.keras.Model):
def __init__(self, params, name=None):
    """Initialize layers to build Geformer model.
    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      name: name of the model.
    """
    super(Geformer, self).__init__(name=name)
    self.params = params
    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        params["vocab_size"], params["hidden_size"])
    self.encoder_stack = EncoderStack(params)
    self.decoder_stack = DecoderStack(params)
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.params["hidden_size"])

  def get_config(self):
    return {
        "params": self.params,
    }

  def call(self, inputs, training):
    """Calculate target logits or inferred target sequences.
    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item (optional), targets: None or int tensor with shape
          [batch_size, target_length].
      training: boolean, whether in training mode or not.
    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          outputs: int tensor with shape [batch_size, decoded_length]
          scores: float tensor with shape [batch_size]}
      Even when float16 is used, the output tensor(s) are always float32.
    Raises:
      NotImplementedError: If try to use padded decode method on CPU/GPUs.
    """
    inputs = inputs if isinstance(inputs, list) else [inputs]
    if len(inputs) == 2:
      inputs, targets = inputs[0], inputs[1]
    else:
      # Decoding path.
      inputs, targets = inputs[0], None
      if self.params["padded_decode"]:
        if not self.params["num_replicas"]:
          raise NotImplementedError(
              "Padded decoding on CPU/GPUs is not supported.")
        decode_batch_size = int(self.params["decode_batch_size"] /
                                self.params["num_replicas"])
        inputs.set_shape([decode_batch_size, self.params["decode_max_length"]])

    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    with tf.name_scope("Transformer"):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = utils.get_padding_bias(inputs)

      # Run the inputs through the encoder layer to map the symbol
      # representations to continuous representations.
      encoder_outputs = self.encode(inputs, attention_bias, training)
      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias, training)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias, training)
        return logits

  def encode(self, inputs, attention_bias, training):
        """Generate continuous representation for inputs.
    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.
    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
    """
    with tf.name_scope("encode"):
      # Prepare inputs to the layer stack by adding positional encodings and
      # applying dropout.
      embedded_inputs = self.embedding_softmax_layer(inputs)
      embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
      inputs_padding = utils.get_padding(inputs)
      attention_bias = tf.cast(attention_bias, self.params["dtype"])

      with tf.name_scope("add_pos_encoding"):
        pos_encoding = self.position_embedding(inputs=embedded_inputs)
        pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
        encoder_inputs = embedded_inputs + pos_encoding

      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params["layer_postprocess_dropout"])

      return self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)
    #denode output: logits for each value in the target sequence, [batch_size, target_length, vocab_size]

  def _get_symbols_to_logits_fn(self, max_decode_length, training):
    """Returns a decoding function that calculates logits of the next tokens."""
    timing_signal = self.position_embedding(
        inputs=None, length=max_decode_length + 1)
    timing_signal = tf.cast(timing_signal, self.params["dtype"])
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length, dtype=self.params["dtype"])

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.
      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1].
        i: Loop index.
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.
      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += timing_signal[i]
      if self.params["padded_decode"]:
        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, i, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
      else:
        self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      decoder_outputs = self.decoder_stack(
          decoder_input,
          cache.get("encoder_outputs"),
          self_attention_bias,
          cache.get("encoder_decoder_attention_bias"),
          training=training,
          cache=cache,
          decode_loop_step=i if self.params["padded_decode"] else None)
      logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache

    return symbols_to_logits_fn

  def predict(self, encoder_outputs, encoder_decoder_attention_bias, training):
    """Return predicted sequence."""
    encoder_outputs = tf.cast(encoder_outputs, self.params["dtype"])
    if self.params["padded_decode"]:
      batch_size = encoder_outputs.shape.as_list()[0]
      input_length = encoder_outputs.shape.as_list()[1]
    else:
      batch_size = tf.shape(encoder_outputs)[0]
      input_length = tf.shape(encoder_outputs)[1]
    max_decode_length = input_length + self.params["extra_decode_length"]
    encoder_decoder_attention_bias = tf.cast(encoder_decoder_attention_bias,
                                             self.params["dtype"])

    symbols_to_logits_fn = self._get_symbols_to_logits_fn(
        max_decode_length, training)

    # Create initial set of IDs that will be passed into symbols_to_logits_fn.
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    # Create cache storing decoder attention values for each layer.
    # pylint: disable=g-complex-comprehension
    init_decode_length = (
        max_decode_length if self.params["padded_decode"] else 0)
    num_heads = self.params["num_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    cache = {
        "layer_%d" % layer: {
            "k":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
                    dtype=self.params["dtype"]),
            "v":
                tf.zeros(
                    [batch_size, init_decode_length, num_heads, dim_per_head],
                    dtype=self.params["dtype"])
        } for layer in range(self.params["num_hidden_layers"])
    }
    # pylint: enable=g-complex-comprehension

    # Add encoder output and attention bias to the cache.
    cache["encoder_outputs"] = encoder_outputs
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    # Use beam search to find the top beam_size sequences and scores.
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=initial_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=max_decode_length,
        eos_id=EOS_ID,
        padded_decode=self.params["padded_decode"],
        dtype=self.params["dtype"])

    # Get the top sequence for each batch element
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]

    return {"outputs": top_decoded_ids, "scores": top_scores}