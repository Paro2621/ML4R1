import numpy as np
import keras as ke

class ML4(ke.Model):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        # ENCODER: Compress data
        self.bottleneck = ke.layers.Dense(h, activation="relu")
        # DECODER: Reconstruct data
        self.output_layer = ke.layers.Dense(784, activation="sigmoid")

    def call(self, inputs):
        x = self.bottleneck(inputs)
        return self.output_layer(x)
    
    def showPattern(self, input):
        pattern = np.array(input).reshape(1, -1)
        return self.output_layer(pattern)
    
    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h})
        return config

class ML4_deep(ke.Model):
    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h
        # ENCODER: Compress data
        self.encoder_hidden = ke.layers.Dense(128, activation="relu")
        self.bottleneck = ke.layers.Dense(h, activation="relu")
        # DECODER: Reconstruct data
        self.decoder_hidden = ke.layers.Dense(128, activation="relu")
        self.output_layer = ke.layers.Dense(784, activation="sigmoid")

    def call(self, inputs):
        x = self.encoder_hidden(inputs)
        x = self.bottleneck(x)
        x = self.decoder_hidden(x)
        return self.output_layer(x)
    
    def showPattern(self, input):
        pattern = np.array(input).reshape(1, -1)
        x = self.decoder_hidden(pattern)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"h": self.h})
        return config
    
if __name__=="__main__":
    pass