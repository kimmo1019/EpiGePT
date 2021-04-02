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

    def call(self, encoder_inputs, attention_bias, training):
        """Return the output of the encoder layer stacks.
        Args:
            encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
            attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
                1, input_length]
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


class ConvStack(tf.keras.layers.Layer):
    """Convolution stack.
    The convolution stack is made up of "num_cb" conv+pooling blocks.
    """
    def __init__(self, params):  
        super(ConvStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        """Builds the convolution stack."""
        params = self.params
        for i in range(params["num_cb"]):
            if i == 0:
                conv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"]//2,
                    kernel_size = (15,1), strides = (1, 1),
                    padding = 'same',activation = None)
                rconv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"]//2,
                    kernel_size = (1,1), strides = (1, 1),
                    padding = 'same')
            else:
                conv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"],
                    kernel_size = (5,1), strides = (1, 1),
                    padding = 'same',activation = None)   
                rconv_layer = tf.keras.layers.Conv2D(
                    filters = params["num_channels"],
                    kernel_size = (1,1), strides = (1, 1),
                    padding = 'same')
            pooling_layer = tf.keras.layers.MaxPool2D(pool_size = (2, 1))
            norm_layer = tf.keras.layers.BatchNormalization()

            self.layers.append([conv_layer, rconv_layer, pooling_layer, norm_layer])

        super(ConvStack, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        """Return the output of the conv stack.
        Args:
            inputs: tensor with shape [batch_size, input_length*2^num_cb, 1, 4]
            training: boolean, whether in training mode or not.
        Returns:
            Output of conv stack.
            float32 tensor with shape [batch_size, input_length, num_channels]
        """
        for i, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            conv_layer, rconv_layer, pooling_layer, norm_layer = layer
            with tf.name_scope("layer_%d" % i):
                x = tf.keras.activations.gelu(inputs) if i == 0 else tf.keras.activations.gelu(x)
                x = conv_layer(x)
                x = pooling_layer(x)
                tmp = rconv_layer(x)
                x = x + tmp
                x = norm_layer(x)
        return tf.squeeze(x, axis = 2)

class Geformer(tf.keras.Model):
    def __init__(self, params, name=None):
        """Initialize layers to build Geformer model.
        Args:
            params: hyperparameter object defining layer sizes, dropout values, etc.
            name: name of the model.
        """
        super(Geformer, self).__init__(name=name)
        self.params = params
        self.conv_stack = ConvStack(params)
        # self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        #         params["vocab_size"], params["hidden_size"])
        self.encoder_stack = EncoderStack(params)
        self.position_embedding = position_embedding.RelativePositionEmbedding(
                hidden_size=self.params["hidden_size"])

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        """Calculate outputs of Geformer model.
        Args:
            inputs: input tensor list of size 1 or 2
                First item: DNA one-hot encoding tensor with shape [batch_size, input_length*2^num_cb, 1, 4].
                Second item: Cell-type-specific embedding with shape [batch_size, input_length, hidden_size].
            training: boolean, whether in training mode or not.
        Returns:
                returns a dictionary {
                    outputs: int tensor with shape [batch_size, decoded_length]
                    scores: float tensor with shape [batch_size]}
            Even when float16 is used, the output tensor(s) are always float32.
        """
        inputs, embedded_celltype = inputs[0], inputs[1]
        # Shape for embedded_celltype [batch_size, input_length, hidden_size - num_channels]

        # Convolution stack for reducing feature size and enhancing receptive field.
        with tf.name_scope("Convolution"):
            conv_outputs = self.conv_stack(inputs)
            # Shape for convolution stack [batch_size, input_length, num_channels]
            embedded_inputs = tf.concat([conv_outputs,embedded_celltype],axis=-1)
            # embedded_inputs with output shape [batch_size, input_length, hidden_size]

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # multi-headed attention layers.
            #attention_bias = utils.get_padding_bias(inputs)
            attention_bias = tf.zeros_like(tf.reduce_mean(embedded_inputs,axis=-1))
            attention_bias = tf.expand_dims(
                tf.expand_dims(attention_bias, axis=1), axis=1)
            encoder_outputs = self.encode(embedded_inputs, attention_bias, training)
            return encoder_outputs

    def encode(self, embedded_inputs, attention_bias, training):
        """Generate continuous representation for inputs.
        Args:
            embedded_inputs: int tensor with shape [batch_size, input_length, hidden_size].
            attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
            training: boolean, whether in training mode or not.
        Returns:
            float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            embedded_inputs = tf.cast(embedded_inputs, self.params["dtype"])
            attention_bias = tf.cast(attention_bias, self.params["dtype"])

            with tf.name_scope("add_pos_encoding"):
                pos_encoding = self.position_embedding(inputs=embedded_inputs)
                pos_encoding = tf.cast(pos_encoding, self.params["dtype"])
                #pos_encoding with shape [input_length, hidden_size], broadcast here.
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                        encoder_inputs, rate=self.params["layer_postprocess_dropout"])

            return self.encoder_stack(
                    encoder_inputs, attention_bias, training=training)