//the required library
const tf = require('@tensorflow/tfjs');

//create a sequential model
const model = tf.sequential();

//add layers
model.add(tf.layers.conv2d({filters:96, kernelSize:11, strides:4, activation: 'relu', inputShape: [224, 224, 3]}));
model.add(tf.layers.batchNormalization());
model.add(tf.layers.maxPooling2d( {poolSize: 2}));
model.add(tf.layers.conv2d({filters:256, kernelSize:5, activation: 'relu'}));
model.add(tf.layers.batchNormalization());
model.add(tf.layers.maxPooling2d( {poolSize: 2}));
model.add(tf.layers.conv2d({filters:384, kernelSize:3, activation: 'relu'}));
model.add(tf.layers.conv2d({filters:384, kernelSize:3, activation: 'relu'}));
model.add(tf.layers.conv2d({filters:256, kernelSize:3, activation: 'relu'}));
model.add(tf.layers.maxPooling2d( {poolSize: 2}));
model.add(tf.layers.dense({units: 4096, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({units: 4096, activation: 'relu'}));
model.add(tf.layers.dropout({rate: 0.5}));
model.add(tf.layers.dense({units: 4096, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

//define an optimizer
const my_opt = tf.train.adam(0.01)

//compile the model
model.compile({optimizer: my_opt, loss: 'meanSquaredError'});

model.summary()
