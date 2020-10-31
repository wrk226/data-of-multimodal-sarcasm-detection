import tensorflow as tf

class ImageFeature():
    def __init__(self):
        self.XList = [tf.compat.v1.placeholder(tf.float32, [None, 2048]) for i in range(196)]
        self.outputLS = [None] * 196
        self.batchSize = tf.compat.v1.placeholder(tf.int32)
        self.defaultFeatureSize = 1024
        with tf.compat.v1.variable_scope("training_variable"):
            self.weights = {
                "FC": tf.compat.v1.get_variable(name="FC", shape=[2048, self.defaultFeatureSize], initializer=tf.initializers.GlorotUniform())
            }
            self.biases = {
                "FC": tf.Variable(tf.constant(0.01, shape=[self.defaultFeatureSize], dtype=tf.float32, name="FC"))
            }
        for i in range(196):
            self.outputLS[i] = tf.nn.relu(tf.matmul(self.XList[i], self.weights["FC"]) + self.biases["FC"])
