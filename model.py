import tensorflow as tf
class Model():
    def __init__(self):
        self.learning_rate = 1
        self.training_epochs = 100
        self.batch_size = 100
        self.display_step = 1
        self.gr_num = 2

    def logistic_regression(self):
        # tf Graph Input
        self.x = tf.placeholder(tf.float32, [None, 15]) # mnist data image of shape 28*28=784
        self.y = tf.placeholder(tf.float32, [None, 3]) # 0-9 digits recognition => 10 classes
        # Set model weights
        self.W = tf.Variable(tf.random_normal([15, 3]))
        self.b = tf.Variable(tf.random_normal([3]))
        # Construct model
        self.pred = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) # Softmax
        # Minimize error using cross entropy
        # self.cost = tf.reduce_mean(self.y*tf.log(self.pred), reduction_indices=1)
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))

        # Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        return self.pred, self.cost

    def FC(self, ch_in, ch_out, input):
        w = tf.Variable(tf.random_normal([ch_in, ch_out]))
        b = tf.Variable(tf.random_normal([ch_out]))
        return tf.nn.relu(tf.matmul(input,w)+b)

    def nn(self):
        # tf Graph Input
        self.x = tf.placeholder(tf.float32, [None, 15]) # mnist data image of shape 28*28=784
        self.y = tf.placeholder(tf.float32, [None, self.gr_num]) # 0-9 digits recognition => 10 classes
        # Set model weights
        self.h1 = self.FC(15,1000, self.x)
        self.h2 = self.FC(1000,10000, self.h1)
        self.h3 = self.FC(10000, 5000, self.h2)
        self.h4 = self.FC(5000, self.gr_num, self.h3)
        # Construct model
        self.pred = tf.nn.softmax(self.h4) # Softmax
        # Minimize error using cross entropy
        # self.cost = tf.reduce_mean(self.y*tf.log(self.pred))
        # self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred))
        # Gradient Descent
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        return self.pred, self.cost

    def session(self,x_tr, y_tr, x_tst, y_tst):
        # Initialize the variables (i.e. assign their default value)
        # Parameters
        init = tf.global_variables_initializer()
        # Start training
        with tf.Session() as sess:
            # Run the initializer
            sess.run(init)
            self.train(sess, x_tr, y_tr)
            self.test(sess, x_tst, y_tst)

    def train(self,sess, x_tr, y_tr):
        # Training cycle
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            # total_batch = int(mnist.train.num_examples/self.batch_size)
            total_batch = len(x_tr)
            # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x:x_tr, self.y:y_tr})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            if (epoch+1) % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"Accuracy:", accuracy.eval({self.x:x_tr, self.y:y_tr}),  "cost=", "{}".format(c))
        print("Optimization Finished!")

    def test(self,sess, x_tst, y_tst):
        # Test model
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({self.x: x_tst, self.y: y_tst}))