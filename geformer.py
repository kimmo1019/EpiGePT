import tensorflow as tf
import utils
from layers import ffn_layer, attention_layer, prepost_layer
from layers import embedding_layer, position_embedding
import numpy as np

"""
To do list
    1. Change module name,  xxxModule, xxx: Transformer, ResConv, MultiTaskPre(done)
    2. Add self-attention weights, need to change attention_layer.py, prepost_layer, geformer(done)
    3. Add dilated convolution to Conv
    4. Add pretrain stage
    5. tf.Tape
    6. Multiple GPU (done)
    7. tf.data.Dataset.from_generator
    8. 1000 parametrized (done)
"""

class TransformerModule(tf.keras.layers.Layer):
    """Transformer module.
    The module is made up of N identical layers. Each layer is composed
    of the sublayers:
        1. Self-attention layer
        2. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, params):  
        super(TransformerModule, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        """Builds the Transformer module."""
        params = self.params
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                    params["hidden_size"], params["num_heads"],
                    params["attention_dropout"])
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                    params["hidden_size"], params["filter_size"], params["relu_dropout"])

            self.layers.append([
                    prepost_layer.PrePostProcessingAttWrapper(self_attention_layer, params),
                    prepost_layer.PrePostProcessingFnnWrapper(feed_forward_network, params)
            ])
        # Create final layer normalization layer.
        self.output_normalization = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32")
        super(TransformerModule, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, attention_bias, training):
        """Return the output of the transformer module.
        Args:
            inputs: tensor with shape [batch_size, input_length, hidden_size]
            attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
                1, input_length]
            training: boolean, whether in training mode or not.
        Returns:
            Outputs of transformer module.
                item[0]: transformer encoded results
                        float32 tensor with shape [batch_size, input_length, hidden_size]
                item[1]: self-attention weights
                        float32 tensor with shape [batch_size, num_hidden_layers, num_heads, input_length, input_length]
        """
        attention_weights = {}
        x = inputs
        for i, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" % i):
                with tf.name_scope("self_attention"):
                    x, w = self_attention_layer(
                            x, attention_bias, training=training)
                    attention_weights['layer_%d' % i] = w
                with tf.name_scope("ffn"):
                    x = feed_forward_network(
                            x, training=training)
        
        #attention_weights = tf.stack(attention_weights, axis = 1)
        return self.output_normalization(x), attention_weights


class ConvModule(tf.keras.layers.Layer):
    """Convolution Module.
    The convolution module is made up of "num_cb" conv+pooling blocks.
    """
    def __init__(self, params):  
        super(ConvModule, self).__init__()
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

        super(ConvModule, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        """Return the output of the convolution module.
        Args:
            inputs: tensor with shape [batch_size, input_length*2^num_cb, 1, 4]
            training: boolean, whether in training mode or not.
        Returns:
            Output of convolution module.
            float32 tensor with shape [batch_size, input_length, num_channels]
        """

        for i, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            conv_layer, rconv_layer, pooling_layer, norm_layer = layer
            with tf.name_scope("layer_%d" % i):
                #x = tf.keras.activations.gelu(inputs) if i == 0 else tf.keras.activations.gelu(x)
                x = tf.keras.activations.relu(inputs) if i == 0 else tf.keras.activations.relu(x)
                x = conv_layer(x)
                x = pooling_layer(x)
                tmp = rconv_layer(x)
                x = x + tmp
                x = norm_layer(x)
        return tf.squeeze(x, axis = 2)

class MultiTaskPreModule(tf.keras.layers.Layer):
    """Multi-task prediction module.
    This module is mainly made up with 1x1 convolutional layers.
    """
    def __init__(self, params):  
        super(MultiTaskPreModule, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        """Builds the Multi-task prediction module."""
        params = self.params
        conv_layer1 = tf.keras.layers.Conv2D(
            filters = 1024,
            kernel_size = (1,1), strides = (1, 1),
            padding = 'same',activation = None)
        norm_layer = tf.keras.layers.BatchNormalization()
        conv_layer2 = tf.keras.layers.Conv2D(
            filters = params['num_targets'],
            kernel_size = (1,1), strides = (1, 1),
            padding = 'same',activation = None)        
        #tf.nn.dropout(weights, rate=self.attention_dropout)

        self.layers = [conv_layer1, norm_layer, conv_layer2]

        super(MultiTaskPreModule, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        """Return the output of the multi-task prediction.
        Args:
            inputs: tensor with shape [batch_size, input_length, hidden_size]
            training: boolean, whether in training mode or not.
        Returns:
            Output of multi-task prediction.
            float32 tensor with shape [batch_size, input_length, num_targets]
        """
        conv_layer1, norm_layer, conv_layer2 = self.layers
        x = tf.expand_dims(inputs, axis=2)
        x = tf.keras.activations.relu(x)
        x = conv_layer1(x)
        x = norm_layer(x)
        x = tf.nn.dropout(x, rate=self.params['attention_dropout'])
        #x = tf.keras.activations.gelu(x)
        x = tf.keras.activations.relu(x)
        x = conv_layer2(x)
        x = tf.math.softplus(x)
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
        self.conv_net = ConvModule(params)
        # self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        #         params["vocab_size"], params["hidden_size"])
        self.position_embedding = position_embedding.RelativePositionEmbedding(
                hidden_size=self.params["hidden_size"])
        self.transformer_net = TransformerModule(params)
        self.multi_task_pre = MultiTaskPreModule(params)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, inputs, training):
        """Calculate outputs of Geformer model.
        Args:
            inputs: input tensor list of size 1 or 2
                First item: DNA one-hot encoding tensor with shape [batch_size, input_length*2^num_cb, 1, 4].
                Second item: Cell-type-specific embedding with shape [batch_size, input_length, hidden_size - num_channels].
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
            conv_outputs = self.conv_net(inputs)
            # Shape for convolution stack [batch_size, input_length, num_channels]
            #embedded_inputs = tf.concat([conv_outputs,embedded_celltype],axis=-1)
            embedded_inputs = tf.keras.layers.Concatenate(axis = -1)([conv_outputs, embedded_celltype])
            # embedded_inputs with output shape [batch_size, input_length, hidden_size]
            # shuffle channels for position embeddings
            np.random.seed(123)
            shuffle_indx = np.random.choice(np.arange(self.params["hidden_size"]),
                           size = self.params["hidden_size"], replace=False)
            embedded_inputs = tf.gather(embedded_inputs, shuffle_indx, axis = -1)

        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        with tf.name_scope("Transformer"):
            # multi-headed attention layers.
            #attention_bias = utils.get_padding_bias(inputs)
            attention_bias = tf.zeros_like(tf.reduce_mean(embedded_inputs,axis=-1))
            attention_bias = tf.expand_dims(
                tf.expand_dims(attention_bias, axis=1), axis=1)
            encoder_outputs, attention_weights = self.encode(embedded_inputs, attention_bias, training)
        with tf.name_scope("Multitask_output"):
            multitask_output = self.multi_task_pre(encoder_outputs)
            return multitask_output, attention_weights
            #return multitask_output
        

    def encode(self, embedded_inputs, attention_bias, training):
        """Encode inputs by transformer module with position embeddings.
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

            return self.transformer_net(
                    encoder_inputs, attention_bias, training=training)