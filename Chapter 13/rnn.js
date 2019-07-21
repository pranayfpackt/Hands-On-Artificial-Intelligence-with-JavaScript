//the required library
const tf = require('@tensorflow/tfjs');

//create a sequential model
const model = tf.sequential();

//add layers
model.add(tf.layers.simpleRNN({units: 100, inputShape: [160,100], returnSequences: true, dropout: 0.5, recurrentDropout: 0.5}));
model.add(tf.layers.simpleRNN({units: 200, dropout: 0.5, recurrentDropout: 0.5}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));

//define an optimizer
const my_opt = tf.train.adam(0.01)

//compile the model
model.compile({optimizer: my_opt, loss: 'meanSquaredError'});

model.summary()

