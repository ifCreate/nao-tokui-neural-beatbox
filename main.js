// keras-based model to classify drum kit sound based on its spectrogram.
// python script: https://gist.github.com/naotokui/a2b331dd206b13a70800e862cfe7da3c
const modelpath = "localized/models/drum_classification_128_augmented/model.json";

// Drum kit
const DRUM_CLASSES = [
   'Kick',
   'Snare',
   'Hi-hat closed',
   'Hi-hat open',
   'Tom low',
   'Tom mid',
   'Tom high',
   'Clap',
   'Rim'
];

const SEGMENT_MIN_LENGTH = 250; // Minimum length of an audio segment
const MAX_REC_DURATION = 10.0; // total recording length in seconds

// Load tensorflow.js model from the path
var isModelLoaded = false;
var isRNNModelLaded = false;
async function loadPretrainedModel() {
   tfmodel = await tf.loadModel(modelpath);
   isModelLoaded = true;
   await startApp();
}
loadPretrainedModel();

// Global
var isReadyToRecord = false;
var isRecording = false;
var isReadyToPlay = false;
var onsets; // segmented regions of recorded audio

var segments = [];
var drumkit_regions = [];

var wavesurfer = WaveSurfer.create({
   container: '#ws-waveform',
   waveColor: 'white',
   progressColor: 'white',
   barWidth: '2',
   scrollParent: true,
   plugins: [
      WaveSurfer.spectrogram.create({
         container: '#ws-spectorogram',
         pixelRatio: 2.0,
      }),
      WaveSurfer.regions.create()
   ]
});
wavesurfer.on('region-click', function (region, e) {
   e.stopPropagation();
   region.play();
});

waveforms_kit = [];
for (let i = 0; i < DRUM_CLASSES.length; i++) {
   let ws = WaveSurfer.create({
      container: '#ws-waveform-kit-' + i.toString(),
      waveColor: 'white',
      progressColor: 'white',
      barWidth: '1',
      plugins: [
         WaveSurfer.regions.create()
      ]
   });
   waveforms_kit.push(ws);

   ws.on('region-click', function (region, e) {
      e.stopPropagation();
      region.play();
   });
}

var mic, recorder, compressor, soundFile;


const TIME_HUMANIZATION = 0.01;

// Add small reverb
let dummySoundPath = 'localized/sound/silent.wav';
let masterComp = new Tone.Compressor().toMaster();

let envelopes = [];
for (let i = 0; i < DRUM_CLASSES.length; i++) {
   var env = new Tone.AmplitudeEnvelope({
      "attack": 0.05,
      "decay": 0.30,
      "sustain": 1.0,
      "release": 0.30,
   });
   env.connect(masterComp);
   envelopes.push(env);
}

// initialize Tone.Players with silent wav file
let drumKit = [];
for (let i = 0; i < DRUM_CLASSES.length; i++) {
   var drum = new Tone.Player(dummySoundPath);
   drum.connect(envelopes[i]);
   drumKit.push(drum);
}

let midiDrums = [36, 38, 42, 46, 41, 43, 45, 49, 51];
let reverseMidiMapping = new Map([ // midi value to drumkit index
   [36, 0],
   [35, 0],
   [38, 1],
   [27, 1],
   [28, 1],
   [31, 1],
   [32, 1],
   [33, 1],
   [34, 1],
   [37, 1],
   [39, 1],
   [40, 1],
   [56, 1],
   [65, 1],
   [66, 1],
   [75, 1],
   [85, 1],
   [42, 2],
   [44, 2],
   [54, 2],
   [68, 2],
   [69, 2],
   [70, 2],
   [71, 2],
   [73, 2],
   [78, 2],
   [80, 2],
   [46, 3],
   [67, 3],
   [72, 3],
   [74, 3],
   [79, 3],
   [81, 3],
   [45, 4],
   [29, 4],
   [41, 4],
   [61, 4],
   [64, 4],
   [84, 4],
   [48, 5],
   [47, 5],
   [60, 5],
   [63, 5],
   [77, 5],
   [86, 5],
   [87, 5],
   [50, 6],
   [30, 6],
   [43, 6],
   [62, 6],
   [76, 6],
   [83, 6],
   [49, 7],
   [55, 7],
   [57, 7],
   [58, 7],
   [51, 8],
   [52, 8],
   [53, 8],
   [59, 8],
   [82, 8]
]);

let temperature = 1.0;

let outputs = {
   internal: {
      play: (drumIdx, velocity, time) => {
         drumKit[drumIdx].start(time);
         envelopes[drumIdx].triggerAttackRelease(0.5, time, velocity);

         if (drumIdx < segments.length) {
            segments[drumIdx].update({
               color: randomColor(0.25)
            });
            drumkit_regions[drumIdx].update({
               color: randomColor(0.25)
            });
         }
      }
   }
};

let rnn = new mm.MusicRNN(
    'localized/models/tfjs_drum_kit_rnn'
);

let seed_input = [[0], [], [2], []];

let state = {
   patternLength: 32,
   seedLength: 4,
   swing: 0.55,
   pattern: seed_input.concat(_.times(28, i => [])),
   tempo: 120
};
let stepEls = [],
    hasBeenStarted = false,
    activeOutput = 'internal';

let making_complex = true; // are we adding more seed notes?
let pattern_seed = seed_input; // original seed
pattern_seed.count = function () {
   let count = 0;
   for (let i = 0; i < pattern_seed.length; i++) {
      count += pattern_seed[i].length
   }
   return count;
}

var isTraining = false;


// Normalize audio buffer to -1 to 1 range
function normalizeBuffer(buffer) {
   var max = 0
   for (var c = 0; c < buffer.numberOfChannels; c++) {
      var data = buffer.getChannelData(c)
      for (var i = 0; i < buffer.length; i++) {
         max = Math.max(Math.abs(data[i]), max)
      }
   }

   var amp = Math.max(1 / max, 1)

   for (var c = 0; c < buffer.numberOfChannels; c++) {
      var data = buffer.getChannelData(c);
      for (var i = 0; i < buffer.length; i++) {
         data[i] = Math.min(Math.max(data[i] * amp, -1), 1);
      }
   }
}

function onLoaded() {
   compressor.process(soundFile);
   processBuffer(soundFile.buffer);
   select('#initialization').hide();
}

function onSoundLoading(progress) {

}

function onFileDropped(file) {
   // If it's an audio file
   if (file.type === 'audio') {
      if (file.size > 3000000) { // 3MB
         alert("Oops... this a file is too big!");
         return;
      }
      select('#initialization').show();
      soundFile = loadSound(file.data, onLoaded, onSoundLoading);
   } else {
      alert("Wrong format!");
   }
   select("#ws-waveform").style('border-color', 'white');
}

function onDragOver() {
   select("#ws-waveform").style('border-color', 'green');
}

function onDragLeave() {
   select("#ws-waveform").style('border-color', 'white');
}

function onRecStop() {
   var waveform = select("#ws-waveform");
   waveform.style('border-color', 'white');

   compressor.process(soundFile);
   normalizeBuffer(soundFile.buffer);
   processBuffer(soundFile.buffer);
   select("#ws-waveform-text").html('');

   isRecording = false;
}

function toggleRecording() {
   getAudioContext().resume();
   mic.start();
   recorder.setInput(mic);
   if (mic.enabled === false) return;

   if (!isRecording) {
      recorder.record(soundFile, MAX_REC_DURATION, onRecStop);
      select("#ws-waveform-text").html('Recording...');

      var waveform = select("#ws-waveform");
      waveform.style('border-color', 'red');
      isRecording = true;
      recStartedAt = millis();
   }
}


function processBuffer(buffer) {
   // Onsets
   // see https://s3-us-west-2.amazonaws.com/s.cdpn.io/2096725/onset.js
   onsets = getOnsets(buffer, SEGMENT_MIN_LENGTH);

   // trim at the first onset
   if (onsets.length > 0) {
      console.log("trim at", onsets[0]);
      buffer = sliceAudioBufferInMono(buffer, onsets[0], buffer.duration);
      onsets = getOnsets(buffer);
   }

   // Show waveform
   wavesurfer.loadDecodedBuffer(buffer);

   // set region
   wavesurfer.clearRegions(); // clear previou data
   segments = [];
   for (var i = 0; i < onsets.length - 1; i++) {
      region = wavesurfer.addRegion({
         id: i,
         start: onsets[i],
         end: onsets[i + 1],
         resize: false,
         drag: false,
         color: randomColor(0.15)
      });
      segments.push(region);
   }
}

function checkVolume(buffer) {
   const AMP_THRESHOLD = 0.1; // does this segment have any sound?
   var array = buffer.getChannelData(0);

   for (let i = 0; i < array.length; i++) {
      if (array[i] > AMP_THRESHOLD) return true;
   }
   return false;
}


function classifyAudioSegment(buffer, startSec, endSec, fftSize = 1024, hopSize = 256, melCount = 128, specLength = 32) {
   // Create audio buffer for the segment
   buffer = sliceAudioBufferInMono(buffer, startSec, endSec);

   // if its too quiet... ignore!
   if (checkVolume(buffer) === false) {
      return _.fill(Array(DRUM_CLASSES.length), 0.0);
   }

   // Get spectrogram matrix
   let db_spectrogram = createSpectrogram(buffer, fftSize, hopSize, melCount, false);

   // Create tf.tensor2d
   // This audio classification model expects spectrograms of [128, 32]  (# of melbanks: 128 / duration: 32 FFT windows)
   const tfbuffer = tf.buffer([melCount, specLength]);

   // Initialize the tfbuffer.  TODO: better initialization??
   for (var i = 0; i < melCount; i++) {
      for (var j = 0; j < specLength; j++) {
         tfbuffer.set(MIN_DB, i, j);
      }
   }

   // Fill the tfbuffer with spectrogram data in dB
   let lng = (db_spectrogram.length < specLength) ? db_spectrogram.length : specLength; // just in case the buffer is shorter than the specified size
   for (var i = 0; i < melCount; i++) {
      for (var j = 0; j < lng; j++) {
         tfbuffer.set(db_spectrogram[j][i], i, j); // cantion: needs to transpose the matrix
      }
   }

   // Reshape for prediction
   input_tensor = tfbuffer.toTensor(); // tf.buffer -> tf.tensor
   input_tensor = tf.reshape(input_tensor, [1, input_tensor.shape[0], input_tensor.shape[1], 1]); // [1, 128, 32, 1]

   // TO DO: fix this loading process
   try {
      let predictions = tfmodel.predict(input_tensor);
      // tfmodel.compile({optimizer: 'adam', loss: 'categoricalCrossentropy',metrics: ['accuracy']});
      // tfmodel.fit(input_tensor, predictions, {epochs: 1, batchSize: 4});
      predictions = predictions.flatten().dataSync(); // tf.tensor -> array
      let predictions_ = [] // we only care the selected set of drums
      for (var i = 0; i < DRUM_CLASSES.length; i++) {
         predictions_.push(predictions[i]);
      }
      return predictions_;
   } catch (err) {
      console.error(err);
      return _.fill(Array(DRUM_CLASSES.length), 0.0);
   }
}

function doesClassifyAll() {
   var predictions = []
   for (var i = 0; i < onsets.length - 1; i++) {
      // Classify the segment
      let prediction = classifyAudioSegment(soundFile.buffer, onsets[i], onsets[i + 1]);
      predictions.push(prediction);
   }
   return predictions;
}

function createDrumKitBuffer(buffer, drumkits, i) {
   if (i >= drumkits.length) return;

   print(DRUM_CLASSES[i], drumkits[i]);
   var index = drumkits[i];
   var startIndex = buffer.sampleRate * onsets[index];
   var endIndex = buffer.sampleRate * onsets[index + 1];
   var tmpArray = buffer.getChannelData(0);
   tmpArray = tmpArray.slice(startIndex, endIndex);
   drumKit[i].buffer.fromArray(tmpArray);

   // show waveform
   let audiobuffer = drumKit[i].buffer.get();
   waveforms_kit[i].loadDecodedBuffer(audiobuffer);
   let drumkit_region = waveforms_kit[i].addRegion({ // react to click event
      id: 0,
      start: 0,
      end: onsets[index + 1] - onsets[index],
      resize: false,
      drag: false
   });
   drumkit_regions[i] = drumkit_region;
}

async function createDrumSet(predictions, allowDuplication = false) {
   var drumkits = []; // array of segment ids

   if (allowDuplication) {
      // create a drum set while allowing duplication = a segment can be used multiple times in a drum kit
      for (let drum in DRUM_CLASSES) {
         let pred_drums = [];
         for (let i = 0; i < predictions.length; i++) {
            pred_drums.push(predictions[i][drum]);
         }
         let selected_id = _.indexOf(pred_drums, _.max(pred_drums));
         drumkits.push(selected_id);
      }
   } else {
      // Create a drum set while avoiding duplication = a segment only used once in a drum kit
      for (let drum in DRUM_CLASSES) {
         let pred_drums = [];
         for (let i = 0; i < predictions.length; i++) {
            pred_drums.push(predictions[i][drum]);
         }

         let pred_drums_sorted = pred_drums.slice(); // copy
         pred_drums_sorted.sort(function (a, b) {
            return b - a
         });
         for (let i = 0; i < pred_drums_sorted.length; i++) {
            let selected_id = _.indexOf(pred_drums, pred_drums_sorted[i]);
            // check if the segment is not selected yet.
            if (!drumkits.includes(selected_id)) {
               drumkits.push(selected_id);
               break;
            }
         }
      }
   }

   // Create audiobuffers
   // FIXME: codepen doesn't like long lasting loops???
   drumkit_regions = [];
   createDrumKitBuffer(soundFile.buffer, drumkits, 0);
   createDrumKitBuffer(soundFile.buffer, drumkits, 1);
   createDrumKitBuffer(soundFile.buffer, drumkits, 2);
   createDrumKitBuffer(soundFile.buffer, drumkits, 3);
   createDrumKitBuffer(soundFile.buffer, drumkits, 4);
   createDrumKitBuffer(soundFile.buffer, drumkits, 5);
   createDrumKitBuffer(soundFile.buffer, drumkits, 6);
   createDrumKitBuffer(soundFile.buffer, drumkits, 7);
   createDrumKitBuffer(soundFile.buffer, drumkits, 8);

   return drumkits;
}

// Analyze button pressed -> classify all audio segments to build a drum kit
async function classifyAll() {
   if (isModelLoaded === false) {
      alert("Error: TensorFlow.js model is not loaded. Check your network connection.");
      return;
   }

   if (!soundFile || soundFile.duration() == 0) {
      alert("You need to record something before analyzing.");
      return;
   }

   // GUI
   select("#progressbar-analysis").show();
   await sleep(100); // dirty hack to reflect the UI change. TODO: fix me!

   // Classification
   var predictions = await doesClassifyAll();

   // Create drumkit based on the predictions
   var drumkits = await createDrumSet(predictions);

   // finished!
   select("#progressbar-analysis").hide();

   isReadyToPlay = true;
}

function doSave(value, type, name) {
   let blob;
   if (typeof window.Blob == "function") {
      blob = new Blob([value], {type: type});
   } else {
      let BlobBuilder = window.BlobBuilder || window.MozBlobBuilder || window.WebKitBlobBuilder || window.MSBlobBuilder;
      let bb = new BlobBuilder();
      bb.append(value);
      blob = bb.getBlob(type);
   }
   let URL = window.URL || window.webkitURL;
   let bloburl = URL.createObjectURL(blob);
   let anchor = document.createElement("a");
   if ('download' in anchor) {
      anchor.style.visibility = "hidden";
      anchor.href = bloburl;
      anchor.download = name;
      document.body.appendChild(anchor);
      let evt = document.createEvent("MouseEvents");
      evt.initEvent("click", true, true);
      anchor.dispatchEvent(evt);
      document.body.removeChild(anchor);
   } else if (navigator.msSaveBlob) {
      navigator.msSaveBlob(blob, name);
   } else {
      location.href = bloburl;
   }
}

function removeByValue(arr, val) {
   for(let i = 0; i < arr.length; i++) {
      if(arr[i] == val) {
         arr.splice(i, 1);
         break;
      }
   }
}

function renderSeedWindow(seed) {
   let seed_parent = document.querySelector('.seed_content').children;
   for(let i = 0; i < seed_parent.length; i++){
      let row_parent = seed_parent[i].children;
      for(let j = 0; j < row_parent.length; j++){
         if(row_parent[j].classList.contains('seed_col')){
            row_parent[j].setAttribute('x', i);
            row_parent[j].setAttribute('y', j - 1);
            if(row_parent[j].classList.contains('seed_col_active')){
               row_parent[j].classList.remove('seed_col_active');
            }
         }
      }
   }
   for(let i = 0; i < seed.length; i++){
      for(let j = 0; j < seed[i].length; j++){
         seed_parent[seed[i][j]].children[i+1].classList.add('seed_col_active');
      }
   }
}


function addChart(id, type, value) {
   let dom = document.getElementById(id);
   dom.innerHTML = '';
   let myChart = echarts.init(dom);
   let app = {};
   option = null;
   option = {
      title: {
         text: type
      },
      tooltip: {
         trigger: 'axis'
      },
      legend: {
         data:[type]
      },
      toolbox: {
         show: false,
      },
      xAxis:  {
         type: 'value',
         axisLabel: {
            formatter: '{value}'
         }
      },
      yAxis: {
         type: 'value',
         axisLabel: {
            formatter: '{value}'
         }
      },
      series: [
         {
            name: type,
            type:'line',
            smooth: true,
            data: value
         }
      ]
   };
   if (option && typeof option === "object") {
      myChart.setOption(option, true);
   }
}


function startApp() {
   select('.spinner').style('display', 'none');
   select('.app').style('display', 'initial');
   getAudioContext().resume();
   mic.start();
   recorder.setInput(mic);
}

function setup() {
   // GUIs
   // select('#start_button').mouseClicked(onClickStart).size(100,30).attribute('disabled','disabled');
   select('#record_button').mouseClicked(toggleRecording).attribute('disabled', 'disabled');
   select('#classify_button').mouseClicked(classifyAll).attribute('disabled', 'disabled');
   select('#play_button').attribute('disabled', 'disabled');
   select('#ws-waveform').drop(onFileDropped); // enable drag and drop of audio files
   select('#ws-waveform').dragOver(onDragOver);
   select('#ws-waveform').dragLeave(onDragLeave);


   document.querySelectorAll('.control').forEach(e => e.addEventListener('mouseenter', evt => {
      let el = evt.target;
      document.querySelector('.info2').style.display = "flex";
      document.querySelector('#' + el.id + 'd').style.opacity = 1;
      document.querySelector('#' + el.id + 'd').style.filter =  "alpha(opacity=" + 100 +")";

   }, false));
   document.querySelectorAll('.control').forEach(e => e.addEventListener('mouseleave', evt => {
      let el = evt.target;
      document.querySelector('#' + el.id + 'd').style.opacity = 0;
      document.querySelector('#' + el.id + 'd').style.filter =  "alpha(opacity=" + 0 +")";
      document.querySelector('.info2').style.display = "none";
   },false));

   $('#pattern-length').on('change', evt => setPatternLength(+evt.target.value)).formSelect();
   document.querySelector('#swing').addEventListener('input', evt => setSwing(+evt.target.value));
   document.querySelector('#tempo').addEventListener('input', evt => setTempo(+evt.target.value));
   document.querySelector('#temperature')
       .addEventListener('input', evt => setTemperature(+evt.target.value));


   // create an audio in and prompt user to allow mic input
   mic = new p5.AudioIn();

   // create a sound recorder and set input
   recorder = new p5.SoundRecorder();

   // compressor - for better audio recording
   compressor = new p5.Compressor();
   compressor.drywet(1.0);
   compressor.threshold(-30);

   // this sound file will be used to playback & save the recording
   soundFile = new p5.SoundFile();
   soundFile.disconnect();
   // soundFile = loadSound("https://dl.dropbox.com/s/00ykku8vjgimnfb/TR-08_KIT_A.wav?raw=1", onLoaded);

   // A plugin for code showing from https://ace.c9.io/
   let editor = ace.edit("code_block");
   editor.setTheme("ace/theme/solarized_dark");
   editor.session.setMode("ace/mode/javascript");

   $.ajax({
      url: 'demo/demo_init.js',
      dataType: 'text',
      success: function(data) {
         editor.insert(data);
      }
   });

   let file_count = 0;
   document.querySelector('#apply_button').addEventListener('click', evt => {
      let el = evt.target;
      let final_code = editor.getValue();
      if(document.querySelector('#trick')){
         document.body.removeChild(document.querySelector('#trick'));
      }
      let trick = document.createElement('script')
      trick.setAttribute('id','#trick');
      trick.innerHTML = final_code;
      document.body.appendChild(trick);
   });
   document.querySelector('#save_button').addEventListener('click', evt => {
      let el = evt.target;
      let final_code = editor.getValue();
      let filename = "demo" + file_count + ".js";
      file_count = file_count + 1;
      doSave(final_code, "text/latex", filename);
   });
   document.querySelector('#load_button').addEventListener('click', evt => {
      let el = evt.target;
      let inputObj=document.createElement('input')
      inputObj.setAttribute('id','_ef');
      inputObj.setAttribute('type','file');
      inputObj.setAttribute("style",'visibility:hidden');
      document.body.appendChild(inputObj);
      inputObj.addEventListener('change', evt => {
         let reader = new FileReader();
         reader.onload = function () {
            let code = this.result;
            editor.setValue(code);
         };
         reader.readAsText(evt.target.files[0]);

      });
      inputObj.click();
      document.body.removeChild(inputObj);
   });
   document.querySelector('#modify_button').addEventListener('click', evt => {
      document.querySelector('.dark_window').style.display='block';
      document.querySelector('.seed_window').style.display='block';
      renderSeedWindow(state.pattern.slice(0,4));
   });
   document.querySelector('#close_seed').addEventListener('click', evt => {
      document.querySelector('.dark_window').style.display='none';
      document.querySelector('.seed_window').style.display='none';
      state.pattern = seed_input.concat(_.times(28, i => []))
   });
   document.querySelectorAll('.seed_col').forEach(e => e.addEventListener('click', evt => {
      let el = evt.target;
      if(!el.classList.contains('seed_col_active')){
         el.classList.add('seed_col_active');
         seed_input[parseInt(el.getAttribute('y'))].push(parseInt(el.getAttribute('x')));
      }
      else{
         el.classList.remove('seed_col_active');
         removeByValue(seed_input[parseInt(el.getAttribute('y'))], parseInt(el.getAttribute('x')))
      }
   },false));
   document.querySelector('#run_button').addEventListener('click', evt => {
      train();
   });
}

function draw() {
   // react to mic input volume
   if (mic.enabled && soundFile.duration() == 0) {
      var level = mic.getLevel();
      select("#ws-waveform").style('background:rgb(' + int(level * 255) + ',0,0)');
   }

   if (isRecording) {
      let elapsed = (millis() - recStartedAt) / 1000.0;
      let percentage = int(elapsed / MAX_REC_DURATION * 100);
      select("#progressbar-record").style('width:' + percentage + '%');
   }

   if (!isReadyToRecord) {
      if (isModelLoaded && isRNNModelLaded) {
         // select('#start_button').removeAttribute('disabled');
         select('#record_button').removeAttribute('disabled');
         select('#classify_button').removeAttribute('disabled');
         select('#play_button').removeAttribute('disabled');
         select('#initialization').hide();
         isReadyToRecord = true;
      }
   }
}



/* UTILITY */

function sleep(ms) {
   return new Promise(resolve => setTimeout(resolve, ms));
}

function toNoteSequence(pattern) {
   return mm.sequences.quantizeNoteSequence({
          ticksPerQuarter: 220,
          totalTime: pattern.length / 2,
          timeSignatures: [{
             time: 0,
             numerator: 4,
             denominator: 4
          }],
          tempos: [{
             time: 0,
             qpm: 120
          }],
          notes: _.flatMap(pattern, (step, index) =>
              step.map(d => ({
                 pitch: midiDrums[d],
                 startTime: index * 0.5,
                 endTime: (index + 1) * 0.5
              }))
          )
       },
       1
   );
}

function fromNoteSequence(seq, patternLength) {
   let res = _.times(patternLength, () => []);
   for (let {
      pitch,
      quantizedStartStep
   } of seq.notes) {
      res[quantizedStartStep].push(reverseMidiMapping.get(pitch));
   }
   return res;
}

function isPlaying() {
   return (Tone.Transport.state === 'started');
}

function playPattern() {
   if (sequence) sequence.dispose();
   sequence = new Tone.Sequence(
       (time, {
          drums,
          stepIdx
       }) => {
          let isSwung = stepIdx % 2 !== 0;
          if (isSwung) {
             time += (state.swing - 0.5) * Tone.Time('8n').toSeconds();
          }
          let velocity = getStepVelocity(stepIdx);
          drums.forEach(d => {
             let humanizedTime = stepIdx === 0 ? time : humanizeTime(time);
             outputs[activeOutput].play(d, velocity, humanizedTime);
          });
       },
       state.pattern.map((drums, stepIdx) => ({
          drums,
          stepIdx
       })),
       '16n'
   );

   Tone.context.resume();
   Tone.Transport.start();
   sequence.start();
}

function generatePattern(seed, length) {
   let seedSeq = toNoteSequence(seed);
   return rnn
       .continueSequence(seedSeq, length, temperature)
       .then(r => seed.concat(fromNoteSequence(r, length)));
}

function regenerate(seed) {
   return generatePattern(seed, state.patternLength - seed.length).then(
       result => {
          state.pattern = result;
       }
   );
}

function getStepVelocity(step) {
   if (step % 4 === 0) {
      return 1.0;
   } else if (step % 2 === 0) {
      return 0.85;
   } else {
      return 0.70;
   }
}

function humanizeTime(time) {
   return time - TIME_HUMANIZATION / 2 + Math.random() * TIME_HUMANIZATION;
}

function updatePattern() {
   sequence.removeAll();
   state.pattern.forEach(function (drums, stepIdx) {
      sequence.at(stepIdx, {
         stepIdx: stepIdx,
         drums: drums
      });
   });
}

function startPlaying() {
   if (isReadyToPlay === false) {
      alert("Your drum kit is not ready! Record and analyze your voice!");
      return;
   }
   // Start playing
   if (!isPlaying()) {
      // Reset the seeds
      pattern_seed = seed_input;

      // Regenerate
      regenerate(pattern_seed).then(() => {
         updatePattern();

         // PLay!
         playPattern();

         select('#play_button').html("3. Pause");
      });
   } else { // stop playing
      Tone.Transport.pause();
      select('#play_button').html("3. Play!!");
   }
}

//////////////////////////////////////////////////////////////////
// The following part is taken from Tero Parviainen's amazing
// Neural Drum Machine
// https://codepen.io/teropa/pen/JLjXGK
// I made a few modifications:
// - added ADSR envelope to each drum sound
// - make the sequence keep contineuously changing

Promise.all([
   rnn.initialize(),
   new Promise(res => Tone.Buffer.on('load', res))
]).then(([vars]) => {
   isRNNModelLaded = true; // set flag

   // GUI
   select('#play_button').mouseClicked(startPlaying);

   // Sequence Object to keep the rhythm track
   sequence = new Tone.Sequence(
      (time, {
         drums,
         stepIdx
      }) => {
         let isSwung = stepIdx % 2 !== 0;
         if (isSwung) {
            time += (state.swing - 0.5) * Tone.Time('8n').toSeconds();
         }
         let velocity = getStepVelocity(stepIdx);
         drums.forEach(d => {
            let humanizedTime = stepIdx === 0 ? time : humanizeTime(time);
            outputs[activeOutput].play(d, velocity, humanizedTime);
         });
      },
      // need to initialize with empty array with the length I wanted to have
      state.pattern.map((drums, stepIdx) => ({
         drums,
         stepIdx
      })),
      '16n'
   );

   // Generate next pattern
   Tone.Transport.scheduleRepeat(function (time) {
      if (isPlaying()) {
         let index = Math.floor(Math.random() * pattern_seed.length);

         if (making_complex) { // first make the seed more complex
            let drumId = Math.floor(Math.random() * DRUM_CLASSES.length);
            if (!pattern_seed[index].includes(drumId)) pattern_seed[index].push(drumId);
            if (pattern_seed.count() > 6) making_complex = false;
         } else { // then less complex.... then loop!
            pattern_seed[index].sort().pop();
            if (pattern_seed.count() <= 3) making_complex = true;
         }
         regenerate(pattern_seed);
      }
   }, "4:0:0", "3:3:0");

   // Update the pattern at the very end of 2 bar loop
   Tone.Transport.scheduleRepeat(function (time) {
      if (isPlaying()) {
         updatePattern();
      }
   }, "4:0:0", "3:3:3");

   // Web MIDI
   // we want to map the keys to drum samples
   const directControlMidi = [80, 81, 82, 96, 97, 98, 112, 113, 114]
   WebMidi.enable(err => {
      if (err) {
         console.error('WebMidi could not be enabled', err);
         return;
      }
      if (!WebMidi.inputs.length) return console.error('WebMidi has no input connected');

      WebMidi.inputs.forEach((input) => {
         console.info('Initialized WebMidi. Connected to ' + input.name);
         input.addListener('noteon', "all", (e) => {
            const drumIndex = directControlMidi.indexOf(e.note.number)
            if (drumIndex === -1) return; // midi note is not linked to a sample.
            drumkit_regions[drumIndex].play();
            drumkit_regions[drumIndex].update({
               color: randomColor(0.25)
            });
         })
      })
      const launchpadOut = WebMidi.getOutputByName('Launchpad Mini')
      if (!launchpadOut) return
      console.info('Lighting up launchpad')
      // turn off existing lights
      for (let note = 0; note < 128; note++) {
         launchpadOut.stopNote(note, 1)
      }
      // light up buttons on launchpad
      directControlMidi.forEach(note => {
         launchpadOut.playNote(note, 1, {
            rawVelocity: true,
            velocity: 124, // green
         })
      })
   })

   renderSeedWindow(state.pattern.slice(0,4));
});



function setPatternLength(newPatternLength) {
   if (newPatternLength < state.patternLength) {
      state.pattern.length = newPatternLength;
   } else {
      for (let i = state.pattern.length; i < newPatternLength; i++) {
         state.pattern.push([]);
      }
   }
   let lengthRatio = newPatternLength / state.patternLength;
   state.seedLength = Math.max(
       1,
       Math.min(newPatternLength - 1, Math.round(state.seedLength * lengthRatio))
   );

   state.patternLength = newPatternLength;
   if (Tone.Transport.state === 'started') {
      // PLay!
      Tone.Transport.pause();
      playPattern();
   }
}

function setTempo(newTempo) {
   Tone.Transport.bpm.value = state.tempo = +newTempo;
}

function setSwing(newSwing) {
   state.swing = newSwing;
   if (Tone.Transport.state === 'started') {
      // PLay!
      Tone.Transport.pause();
      playPattern();
   }
}

function setTemperature(newTemperature) {
   temperature = newTemperature;
   Tone.Transport.pause();
   startPlaying();
}

function setSeed(newSeed) {
   state.pattern = newSeed.concat(_.times(28, i => []));
   seed_input = newSeed;
}

function train(){
   document.querySelectorAll('.load_img').forEach(e => {
      e.style.display = 'block';
   });
   document.querySelectorAll('.load_text').forEach(e => {
      e.style.display = 'block';
   });
   if(!isTraining){
      isTraining = true;
      $.ajax({
         url: 'http://localhost:5000/train',
         success: function(data) {
            data = $.parseJSON(data);
            let epochs = data['loss'].length;
            let loss = new Array(epochs);
            let acc = new Array(epochs);
            for(let i = 0; i < loss.length; i++){
               loss[i] = new Array(2);
               loss[i][0] = i;
               loss[i][1] = data['loss'][i]
               acc[i] = new Array(2);
               acc[i][0] = i;
               acc[i][1] = data['acc'][i]
            }

            // console.log(loss, acc);
            // console.log(data);
            addChart('loss', 'Loss', loss);
            addChart('acc', 'Acc', acc);
            isTraining = false;
            alert('Training success!');
         },
         error: function(msg) {
            alert('Some thing wrong with the training module!');
         }
      });
   }


   // $.ajax({
   //    url: 'demo/data.json',
   //    success: function(data) {
   //       let train_total_num = 4;
   //       let train_total_drums = 9;
   //       let melCount = 128;
   //       let specLength = 128;
   //       // Create tf.tensor2d
   //       // This audio classification model expects spectrograms of [128, 32]  (# of melbanks: 128 / duration: 32 FFT windows)
   //       const tfbuffer1 = tf.buffer([train_total_num, melCount, specLength]);
   //
   //       // Initialize the tfbuffer.  TODO: better initialization??
   //       for(var a = 0; a < train_total_num; a++){
   //          for (var i = 0; i < melCount; i++) {
   //             for (var j = 0; j < specLength; j++) {
   //                tfbuffer1.set(MIN_DB, a, i, j);
   //             }
   //          }
   //       }
   //       for(var a = 0; a < train_total_num; a++) {
   //          for (var i = 0; i < melCount; i++) {
   //             for (var j = 0; j < specLength; j++) {
   //                tfbuffer1.set(data['train_melspecs'][a][j][i], a, i, j); // cantion: needs to transpose the matrix
   //             }
   //          }
   //       }
   //
   //       // Reshape for prediction
   //       train_tensor = tfbuffer1.toTensor(); // tf.buffer -> tf.tensor
   //       train_tensor = tf.reshape(train_tensor, [train_tensor.shape[0], train_tensor.shape[1], train_tensor.shape[2], 1]); // [40, 128, 128, 1]
   //
   //
   //       const tfbuffer2 = tf.buffer([train_total_num, train_total_drums]);
   //
   //       // Initialize the tfbuffer.  TODO: better initialization??
   //       for (var i = 0; i < train_total_num; i++) {
   //          for (var j = 0; j < train_total_drums; j++) {
   //             tfbuffer2.set(MIN_DB, i, j);
   //          }
   //       }
   //
   //       let labels = [];
   //       for (var i = 0; i < train_total_num; i++) {
   //          labels.push(data['train_genres'][i].concat(_.times(36, t => 0)))
   //       }
   //       for (var i = 0; i < train_total_num; i++) {
   //          for (var j = 0; j < train_total_drums; j++) {
   //             tfbuffer2.set(labels[i][j], i, j); // cantion: needs to transpose the matrix
   //          }
   //       }
   //
   //       // Reshape for prediction
   //       label_tensor = tfbuffer2.toTensor(); // tf.buffer -> tf.tensor
   //       label_tensor = tf.reshape(label_tensor, [label_tensor.shape[0], label_tensor.shape[1]]); // [40, 45]
   //
   //       let model1;
   //       async function load_model1() {
   //          model1 = await tf.loadModel('localized/models/user_model/model.json');
   //          await model1.summary();
   //          await train_data(model1);
   //       }
   //
   //
   //       async function train_data(model){
   //          model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy',metrics: ['accuracy']});
   //          let p = model.predict(train_tensor);
   //
   //          // console.log(train_tensor, p);
   //          let hist = await model.fit(train_tensor, p,
   //              {
   //                 epochs: 1
   //              });
   //          await console.log(hist);
   //       }
   //
   //       load_model1();
   //    }
   //
   // });
}