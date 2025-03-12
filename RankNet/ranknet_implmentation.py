import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class RankNet:
    def __init__(self, input_shape, hidden_units=[64, 32]):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.model = self.build_model()

    def build_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = input_layer
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu')(x)
        output_layer = layers.Dense(1, activation='linear')(x)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.compile(optimizer='adam', loss=self.ranknet_loss)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def ranknet_loss(self, y_true, y_pred):
        # y_true: true relevance scores (not used directly in RankNet loss)
        # y_pred: predicted scores for pairs of documents

        # Split the predictions into two parts: s_i and s_j
        s_i = y_pred[:, 0]
        s_j = y_pred[:, 1]

        # Calculate the difference between the scores
        s_diff = s_i - s_j

        # Calculate the RankNet loss
        loss = tf.math.log(1 + tf.math.exp(-s_diff))
        return tf.reduce_mean(loss)

# Example usage
if __name__ == "__main__":
    # Example data: X_train is the feature matrix, y_train is the relevance scores
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features each
    y_train = np.random.randint(0, 5, size=(100,))  # Relevance scores (0-4)

    # Create pairs of documents and their relevance differences
    pairs = []
    for i in range(len(X_train)):
        for j in range(len(X_train)):
            if y_train[i] > y_train[j]:
                pairs.append((X_train[i], X_train[j], 1))  # i is more relevant than j
            elif y_train[i] < y_train[j]:
                pairs.append((X_train[i], X_train[j], -1))  # j is more relevant than i

    # Convert pairs to training data
    X_pairs = np.array([np.concatenate([pair[0], pair[1]]) for pair in pairs])
    y_pairs = np.array([pair[2] for pair in pairs])

    # Initialize and train RankNet
    ranknet = RankNet(input_shape=(X_train.shape[1] * 2,))
    ranknet.train(X_pairs, y_pairs, epochs=10, batch_size=32)

    # Predict rankings for new data
    X_test = np.random.rand(10, 10)  # 10 new samples
    predictions = ranknet.predict(X_test)
    print("Predictions:", predictions)