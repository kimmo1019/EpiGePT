import tensorflow as tf

class PrePostProcessingFnnWrapper(tf.keras.layers.Layer):
    """Wrapper class for Fnn that applies layer pre-processing and post-processing."""
    def __init__(self, layer, params):
        super(PrePostProcessingFnnWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32")
        super(PrePostProcessingFnnWrapper, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y

class PrePostProcessingAttWrapper(tf.keras.layers.Layer):
    """Wrapper class for Attention that applies layer pre-processing and post-processing."""
    def __init__(self, layer, params):
        super(PrePostProcessingAttWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32")
        super(PrePostProcessingAttWrapper, self).build(input_shape)

    def get_config(self):
        return {
                "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y, w = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y, w