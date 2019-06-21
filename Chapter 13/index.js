//the required library
const tf = require('@tensorflow/tfjs');

//create a sequential model
const model = tf.sequential();

//add layers
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));

//compile the model
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

//fit the model
const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`);
  }
});