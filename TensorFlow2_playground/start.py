import tensorflow as tf

mnist = tf.keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_test.shape, x_test.shape)

#Sequential: Adding multiple Layers togehter 
#flatten layer: 2D input, 1D output, takes images and retunts a simple int
#Dense Layer: connects every neuron with pre-neurons :w
    # activation : Activation function like Relu to check boundaries 
#Dropout Layer: 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(150, activation='relu'),
    #128 -> Number of Nodes, with activation function rectifier(Sprung) 
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])


model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)


test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)


prob = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
    ])

print(x_test)







