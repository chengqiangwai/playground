import * as nn from "./nn";
import { HeatMap, reduceMatrix } from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from "./state";
import { Example2D, shuffle } from "./dataset";
import { AppendingLineChart } from "./linechart";
import * as d3 from 'd3';
import * as echarts from 'echarts';

let mainWidth;

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;
const MAX_ITER = 500;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: { [name: string]: InputFeature } = {
  "x": { f: (x, y) => x, label: "X_1" },
  "y": { f: (x, y) => y, label: "X_2" },
  "xSquared": { f: (x, y) => x * x, label: "X_1^2" },
  "ySquared": { f: (x, y) => y * y, label: "X_2^2" },
  "xTimesY": { f: (x, y) => x * y, label: "X_1X_2" },
  "sinX": { f: (x, y) => Math.sin(x), label: "sin(X_1)" },
  "sinY": { f: (x, y) => Math.sin(y), label: "sin(X_2)" },
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

// 一些颜色值设置
const BACKGROUND_COLOR = new echarts.graphic.RadialGradient(0.3, 0.3, 0.8, [{
  offset: 0,
  color: '#f7f8fa'
}, {
  offset: 1,
  color: '#cdd0d5'
}])
const GOOD_COLOR = new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
  offset: 0,
  color: 'rgb(129, 227, 238)'
}, {
  offset: 1,
  color: 'rgb(25, 183, 207)'
}]);
const BAD_COLOR = new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
  offset: 0,
  color: 'rgb(251, 118, 123)'
}, {
  offset: 1,
  color: 'rgb(204, 46, 72)'
}]);
const GOOD_SHADOW_COLOR = 'rgba(120, 36, 50, 0.5)';
const BAD_SHADOW_COLOR = 'rgba(25, 100, 150, 0.5)';

// echart的相关配置
let echartContainer = document.getElementById("echartContainer");
let myChart = echarts.init(echartContainer);
window.addEventListener("resize", function () {
  myChart.resize();
});
let echartOption = {
  backgroundColor: BACKGROUND_COLOR,
  title: { text: 'BPNN' },
  grid: [
    { width: '55%', height: '90%', top: 30, left: 30, },
    { width: '30%', height: '20%', top: 30, right: 30, },
    { width: '30%', height: '65%', top: '30%', right: 30 },
  ],
  legend: {
    left: '50%',
  },
  trigger: 'item',
  xAxis: [
    { gridIndex: 0, },
    { gridIndex: 1, min: 1 },
    { gridIndex: 2, },
  ],
  yAxis: [
    { gridIndex: 0, },
    { gridIndex: 1, },
    { gridIndex: 2, },
  ],
  series: []
};
let seriesInOption = [];
function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}




let lossProcess = {
  lossTest: [],
  lossTrain: []
};
function reset(onStartup = false) {
  state.serialize();
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0, 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
    nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
    state.regularization, constructInputIds(), state.initZero);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  lossProcess.lossTest.push([iter, lossTest]);
  lossProcess.lossTrain.push([iter, lossTrain]);
};

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        // simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: { [id: string]: number[][] } = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];

let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}



export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}


function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    // userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
    NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
    state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  // heatMap.updatePoints(trainData);
  // heatMap.updateTestPoints(state.showTestData ? testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function oneStep(): { iter: number, lossTest: number, lossTrain: number } {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });

  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  return {
    iter: iter,
    lossTrain: lossTrain,
    lossTest: lossTest
  }
  // drawLoss({
  //   iter: iter,
  //   lossTrain: lossTrain,
  //   lossTest: lossTest
  // } )
  // if( iter < MAX_ITER){
  //   requestAnimationFrame(oneStep);
  // }
  // updateUI();
}

function runTrain() {
  let tempLoss = oneStep();
  lossProcess.lossTest.push([tempLoss.iter, tempLoss.lossTest]);
  lossProcess.lossTrain.push([tempLoss.iter, tempLoss.lossTrain]);
  drawLoss(lossProcess);

  if (iter < MAX_ITER) {
    setTimeout(() => {
      runTrain();
    }, 100)
  }

  // while (runTimes--) {
  //   let tempLoss = oneStep();
  //   lossProcess.lossTest.push([tempLoss.iter, tempLoss.lossTest]);
  //   lossProcess.lossTrain.push([tempLoss.iter, tempLoss.lossTrain]);
  //   drawLoss(lossProcess);
  // }
}

// 绘制损失函数
function drawLoss(lossProcess) {
  seriesInOption[1] = {
    id: "lossTest",
    type: "line",
    smooth: true,
    xAxisIndex: 1,
    yAxisIndex: 1,
    data: lossProcess.lossTest,
    emphasis: {
      label: {
        show: true,
        position: 'top',
        distance: 15,
        formatter: (params) => {
          let data = params.data;
          return `Iter:${data[0]}\n\nTrain loss: ${data[1].toFixed(4)}`
        },
        color: 'blue',
        align: 'left'
      }
    },
    tooltip: {
      formatter: 'you are beautiful'
    }
  };
  seriesInOption[2] = {
    id: "lossTrain",
    type: "line",
    smooth: true,
    xAxisIndex: 1,
    yAxisIndex: 1,
    emphasis: {
      label: {
        show: true,
        position: 'top',
        distance: 15,
        formatter: (params, i, j, k) => {
          let data = params.data;
          return `Iter:${data[0]}\n\nTrain loss: ${data[1].toFixed(4)}`
        },
        color: 'blue',
        align: 'left'
      }
    },
    label: {
      show: true,
        position: [-100, -50],
        distance: 15,
        formatter: (params) => {
          let data = params.data;
          console.log('hi')
          if(params.dataIndex === lossProcess.lossTrain.length -1) {
            return `Iter:${data[0]}\n\nTrain loss: ${data[1].toFixed(4)}`
          }else{
            return ''
          }
        },
        color: 'blue',
        align: 'left'
    },
    data: lossProcess.lossTrain
  }

  let twoPoint = [
    [iter, lossProcess.lossTest[0]]
  ]
  seriesInOption[3] = {
    id: 'lossLegend',
    type: 'scatter',
    xAxisIndex: 1,
    yAxisIndex: 1,
  }
  echartOption.series = seriesInOption;
  myChart.setOption(echartOption);
}
function drawSamples(samples) {
  let data = samples.map((item) => {
    return [item.x, item.y, item.label]
  })
  seriesInOption.push({
    id: 'samples scatter',
    type: "scatter",
    xAxisIndex: 2,
    yAxisIndex: 2,
    data: data,
    itemStyle: {
      shadowBlur: 10,
      shadowColor: BAD_SHADOW_COLOR,
      shadowOffsetY: 5,
      color: (params) => {
        if (params.data[2] === 1) {
          return GOOD_COLOR
        } else {
          return BAD_COLOR
        }
      }
    },
    symbolSize: 12,
    emphasis: {
      label: {
        show: true,
        position: 'top',
        distance: 15,
        formatter: (params) => {
          let data = params.data;
          return `point: (${data[0].toFixed(3)}, ${data[1].toFixed(3)})\n\nlabel: ${data[2]}`
        },
        color: 'blue',
        align: 'left'
      }
    }
  })
  echartOption.series = seriesInOption;
  myChart.setOption(echartOption);
}
reset(true);
generateData(true);
drawSamples(trainData);
oneStep();
runTrain();
// drawDatasetThumbnails();
// initTutorial();
// makeGUI();
// generateData(true);
// reset(true);
// hideControls();
