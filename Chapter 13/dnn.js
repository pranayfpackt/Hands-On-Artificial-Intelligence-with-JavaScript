//the required library
const tf = require('@tensorflow/tfjs'); 

//create a sequential model
const model = tf.sequential();

//add layers
model.add(tf.layers.dense({units: 10, activation: 'relu', inputShape: [13]}));
model.add(tf.layers.dense({units: 10, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));

//define an optimizer
const my_opt = tf.train.adam(0.01)

//compile the model
model.compile({optimizer: my_opt, loss: 'meanSquaredError'});

model.summary()

//import data
const boston = require('./boston/train.json');

// Map the trainingdata
const trainingData = tf.tensor2d(boston.map(item=> [
  item.crim,
  item.zn,
  item.indus,
  item.chas,
  item.nox,
  item.rm,
  item.age,
  item.dis,
  item.rad,
  item.tax,
  item.ptratio,
  item.black,
  item.lstat
]
),[333,13])

// Map the labels
const outputData = tf.tensor2d(boston.map(item => [
  item.medv
]), [333,1])

//fit the model
async function train_model() {
  let res = await model.fit(trainingData, outputData, {epochs: 300, callbacks: {
    onEpochEnd: (epoch, log) => console.dir(`Epoch ${epoch}: loss = ${log.loss}`)
  }});
}


async function main(){
  await train_model();
  console.log('-- Model Training Complete --');
  const testData = tf.tensor2d([[0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,392.83,4.03]]);
  model.predict(testData).print();
}

main();
