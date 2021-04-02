import tensorflow as tf

class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""
    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype="float32")
        super(PrePostProcessingWrapper, self).build(input_shape)

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