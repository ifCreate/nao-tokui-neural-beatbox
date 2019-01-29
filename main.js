
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
wavesurfer.on('region-click', function(region, e) {
   e.stopPropagation();
   region.play();
});

waveforms_kit = [];
for (let i=0; i<DRUM_CLASSES.length; i++){
   let ws = WaveSurfer.create({
      container: '#ws-waveform-kit-'+i.toString(),
      waveColor: 'white',
      progressColor: 'white',
      barWidth: '1',   
      plugins: [
         WaveSurfer.regions.create()
      ]
   });
   waveforms_kit.push(ws);

   ws.on('region-click', function(region, e) {
      e.stopPropagation();
      region.play();
   });
}

var mic, recorder, compressor, soundFile;

function onClickStart() {
    // start audio context & mic.
    // Chrome requires that audio context must start after user input
    getAudioContext().resume();
    mic.start();
    recorder.setInput(mic);

    // display the rest of the page
    select('.app').style('display', 'initial');
    // hide the start button itself
    select('#start_button')
}

function setup() {
   // GUIs
   select('#start_button').mouseClicked(onClickStart).size(100,50).attribute('disabled','disabled');
   select('#record_button').mouseClicked(toggleRecording).size(100,50).attribute('disabled','disabled');
   select('#classify_button').mouseClicked(classifyAll).size(100,50).attribute('disabled','disabled');
   select('#play_button').size(100,50).attribute('disabled', 'disabled');
   select('#ws-waveform').drop(onFileDropped); // enable drag and drop of audio files
   select('#ws-waveform').dragOver(onDragOver); 
   select('#ws-waveform').dragLeave(onDragLeave);

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

}


// Reset the seed and start generating rhythms!
function startPlaying(){
   // reset the seeds

}

// Analyze button pressed -> classify all audio segments to build a drum kit
async function classifyAll(){
   if (isModelLoaded === false){
      alert("Error: TensorFlow.js model is not loaded. Check your network connection.");
      return;
   }

   if (!soundFile || soundFile.duration() == 0){
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

async function createDrumSet(predictions, allowDuplication = false){
   var drumkits = []; // array of segment ids

   if (allowDuplication){
      // create a drum set while allowing duplication = a segment can be used multiple times in a drum kit
      for (let drum in DRUM_CLASSES){
         let pred_drums = [];
         for (let i = 0; i < predictions.length; i++){
            pred_drums.push(predictions[i][drum]);
         }       
         let selected_id = _.indexOf(pred_drums, _.max(pred_drums));
         drumkits.push(selected_id);   
      }
   }else{
      // Create a drum set while avoiding duplication = a segment only used once in a drum kit
      for (let drum in DRUM_CLASSES){
         let pred_drums = [];
         for (let i = 0; i < predictions.length; i++){
            pred_drums.push(predictions[i][drum]);
         }         

         let pred_drums_sorted = pred_drums.slice(); // copy
         pred_drums_sorted.sort(function(a, b){return b - a});
         for (let i =0; i < pred_drums_sorted.length; i++){ 
            let selected_id = _.indexOf(pred_drums, pred_drums_sorted[i]);
            // check if the segment is not selected yet.
            if (!drumkits.includes(selected_id)){ 
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

function createDrumKitBuffer(buffer, drumkits, i){
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
   let drumkit_region = waveforms_kit[i].addRegion({  // react to click event
      id: 0,
      start: 0, 
      end: onsets[index+1] - onsets[index],
      resize: false,
      drag: false
   });
   drumkit_regions[i] = drumkit_region;
}

function doesClassifyAll(){
   var predictions = []
   for (var i = 0; i < onsets.length-1; i++) {    
      // Classify the segment
      let prediction = classifyAudioSegment(soundFile.buffer, onsets[i], onsets[i+1]);
      predictions.push(prediction);
   }   
   return predictions;
}

// Normalize audio buffer to -1 to 1 range
function normalizeBuffer (buffer) {
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

function onLoaded(){
   compressor.process(soundFile);
   processBuffer(soundFile.buffer);
   select('#initialization').hide();
}

function onSoundLoading(progress){
   
}

function onFileDropped(file){
   // If it's an audio file
   if (file.type === 'audio') {
      if (file.size > 3000000){ // 3MB
         alert("Oops... this a file is too big!");        
         return;
      }       
      select('#initialization').show();
      soundFile = loadSound(file.data, onLoaded,  onSoundLoading);
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

function onRecStop(){
   var waveform = select("#ws-waveform");
   waveform.style('border-color', 'white');

   compressor.process(soundFile);
   normalizeBuffer(soundFile.buffer);
   processBuffer(soundFile.buffer);
   select("#ws-waveform-text").html('');  

   isRecording = false;
}

function toggleRecording(){
   if (mic.enabled === false) return;

   if (!isRecording){   
      recorder.record(soundFile, MAX_REC_DURATION, onRecStop);
      select("#ws-waveform-text").html('Recording...');

      var waveform = select("#ws-waveform");
      waveform.style('border-color', 'red');
      isRecording = true;
      recStartedAt = millis();
   } 
}

function draw(){
   // react to mic input volume
   if (mic.enabled && soundFile.duration() == 0){
      var level = mic.getLevel();
      select("#ws-waveform").style('background:rgb('+int(level* 255)+',0,0)');
   }

   if (isRecording){
      let elapsed = (millis() - recStartedAt)/1000.0;
      let percentage = int(elapsed / MAX_REC_DURATION * 100);
      select("#progressbar-record").style('width:'+percentage+'%');
   }

   if (!isReadyToRecord){
      if (isModelLoaded && isRNNModelLaded){
         select('#start_button').removeAttribute('disabled');
         select('#record_button').removeAttribute('disabled');
         select('#classify_button').removeAttribute('disabled');
         select('#play_button').removeAttribute('disabled');
         select('#initialization').hide();
         isReadyToRecord = true;
      }
   }
}

function processBuffer(buffer){
   // Onsets
   // see https://s3-us-west-2.amazonaws.com/s.cdpn.io/2096725/onset.js
   onsets = getOnsets(buffer, SEGMENT_MIN_LENGTH);    

   // trim at the first onset
   if (onsets.length > 0){      
      console.log("trim at", onsets[0]);      
      buffer = sliceAudioBufferInMono(buffer, onsets[0], buffer.duration);
      onsets = getOnsets(buffer);   
   }

   // Show waveform
   wavesurfer.loadDecodedBuffer(buffer);

   // set region
   wavesurfer.clearRegions(); // clear previou data
   segments = [];
   for (var i = 0; i < onsets.length-1; i++) {    
      region = wavesurfer.addRegion({
         id: i,
         start: onsets[i],
         end: onsets[i+1],
         resize: false,
         drag: false,
         color: randomColor(0.15)
      });
      segments.push(region);
   }
}

function checkVolume(buffer){
   const AMP_THRESHOLD = 0.1; // does this segment have any sound?
   var array = buffer.getChannelData(0);

   for (let i=0; i<array.length; i++){
      if (array[i] > AMP_THRESHOLD) return true;   
   }
   return false;
}

function classifyAudioSegment(buffer, startSec, endSec, fftSize=1024, hopSize=256, melCount=128, specLength=32){
   // Create audio buffer for the segment  
   buffer = sliceAudioBufferInMono(buffer, startSec, endSec);

   // if its too quiet... ignore!
   if (checkVolume(buffer) === false){
      return _.fill(Array(DRUM_CLASSES.length), 0.0);
   }

   // Get spectrogram matrix
   let db_spectrogram = createSpectrogram(buffer, fftSize, hopSize, melCount, false); 

   // Create tf.tensor2d
   // This audio classification model expects spectrograms of [128, 32]  (# of melbanks: 128 / duration: 32 FFT windows) 
   const tfbuffer = tf.buffer([melCount, specLength]);

   // Initialize the tfbuffer.  TODO: better initialization??
   for (var i = 0; i < melCount ; i++) {
      for (var j = 0; j < specLength; j++) {
         tfbuffer.set(MIN_DB, i, j);  
      }
   }

   // Fill the tfbuffer with spectrogram data in dB
   let lng = (db_spectrogram.length < specLength)? db_spectrogram.length : specLength; // just in case the buffer is shorter than the specified size
   for (var i = 0; i < melCount ; i++) {
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
      predictions =  predictions.flatten().dataSync(); // tf.tensor -> array
      let predictions_ = [] // we only care the selected set of drums
      for (var i =0; i < DRUM_CLASSES.length; i++){
         predictions_.push(predictions[i]);
      }
      return predictions_;
   } catch( err ) {
      console.error( err );
      return _.fill(Array(DRUM_CLASSES.length), 0.0);
   }
}


/* UTILITY */

function sleep(ms) {
   return new Promise(resolve => setTimeout(resolve, ms));
}

//////////////////////////////////////////////////////////////////
// The following part is taken from Tero Parviainen's amazing 
// Neural Drum Machine 
// https://codepen.io/teropa/pen/JLjXGK
// I made a few modifications:
// - added ADSR envelope to each drum sound
// - make the sequence keep contineuously changing 

const TIME_HUMANIZATION = 0.01;

// Add small reverb
let dummySoundPath = 'localized/sound/silent.wav';
let masterComp = new Tone.Compressor().toMaster();

let envelopes = [];
for (let i=0; i < DRUM_CLASSES.length; i++){
   var env = new Tone.AmplitudeEnvelope({
      "attack" : 0.05,
      "decay" : 0.30,
      "sustain" : 1.0,
      "release" : 0.30,
   });
   env.connect(masterComp);
   envelopes.push(env);   
}

// let gains = [];
// for (let i=0; i < DRUM_CLASSES.length; i++){
//    var gain = new Tone.Gain();
//    envelopes[i].connect(gain.gain);
//    gain.gain = 0.0;
//    gain.connect(masterComp);
//    gains.push(gain);   
// }

// initialize Tone.Players with silent wav file
let drumKit = [];
for (let i=0; i < DRUM_CLASSES.length; i++){
   var drum = new Tone.Player(dummySoundPath);
   drum.connect(envelopes[i]);
   drumKit.push(drum);
}

let midiDrums = [36, 38, 42, 46, 41, 43, 45, 49, 51];
let reverseMidiMapping = new Map([  // midi value to drumkit index
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
         envelopes[drumIdx].triggerAttackRelease (0.5, time, velocity);
        
         if (drumIdx < segments.length){
            segments[drumIdx].update({color:randomColor(0.25)});
            drumkit_regions[drumIdx].update({color:randomColor(0.25)});
         }
      }
   }
};

let rnn = new mm.MusicRNN(
   'localized/models/tfjs_drum_kit_rnn'
);

Promise.all([
   rnn.initialize(),
   new Promise(res => Tone.Buffer.on('load', res))
]).then(([vars]) => {
   isRNNModelLaded = true; // set flag
   let state = {
      patternLength: 32,
      seedLength: 4,
      swing: 0.55,
      pattern: [[0], [], [2], []].concat(_.times(28, i => [])),
      tempo: 120
   };
   let stepEls = [],
       hasBeenStarted = false,
       activeOutput = 'internal';

   // GUI
   select('#play_button').mouseClicked(startPlaying);

   function isPlaying(){
      return (Tone.Transport.state === 'started');
   }

   // Sequence Object to keep the rhythm track
   sequence = new Tone.Sequence(
      (time, { drums, stepIdx }) => {      
         let isSwung = stepIdx % 2 !== 0;
         if (isSwung) {
            time += (state.swing - 0.5) * Tone.Time('8n').toSeconds();
         }
         let velocity = getStepVelocity(stepIdx);
         drums.forEach(d => {
            let humanizedTime = stepIdx === 0 ? time : humanizeTime(time);
            outputs[activeOutput].play(d, velocity, humanizedTime);
            visualizePlay(humanizedTime, stepIdx, d);
         });
      },
      // need to initialize with empty array with the length I wanted to have
      state.pattern.map((drums, stepIdx) => ({ drums, stepIdx})),
      '16n'
   );

   const original_seed = [[0], [], [2], []]; 
   let making_complex = true; // are we adding more seed notes?
   let pattern_seed = original_seed; // original seed
   pattern_seed.count = function(){
      let count = 0;
      for (let i =0;  i< pattern_seed.length; i++){
         count += pattern_seed[i].length
      }
      return count;
   } 


   function startPlaying(){
      if (isReadyToPlay === false){
         alert("Your drum kit is not ready! Record and analyze your voice!");
         return;
      }
      // Start playing
      if (!isPlaying()){
         // Reset the seeds
         pattern_seed = original_seed;

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


   // Generate next pattern
   Tone.Transport.scheduleRepeat(function(time){
      if (isPlaying()) {
         let index = Math.floor(Math.random() * pattern_seed.length);

         if (making_complex){ // first make the seed more complex
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
   Tone.Transport.scheduleRepeat(function(time){
      if (isPlaying()) {
         updatePattern();
      }
   }, "4:0:0", "3:3:3");

   function generatePattern(seed, length) {
      let seedSeq = toNoteSequence(seed);
      return rnn
         .continueSequence(seedSeq, length, temperature)
         .then(r => seed.concat(fromNoteSequence(r, length)));
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

   function playPattern() {
      if (sequence) sequence.dispose();
      sequence = new Tone.Sequence(
         (time, { drums, stepIdx }) => {
            let isSwung = stepIdx % 2 !== 0;
            if (isSwung) {
               time += (state.swing - 0.5) * Tone.Time('8n').toSeconds();
            }
            let velocity = getStepVelocity(stepIdx);
            drums.forEach(d => {
               let humanizedTime = stepIdx === 0 ? time : humanizeTime(time);
               outputs[activeOutput].play(d, velocity, humanizedTime);
               // visualizePlay(humanizedTime, stepIdx, d);
            });
         },
         state.pattern.map((drums, stepIdx) => ({ drums, stepIdx })),
         '16n'
      );
      
      Tone.context.resume();
      Tone.Transport.start(); 
      sequence.start();
   }

   function visualizePlay(time, stepIdx, drumIdx) {
      // Tone.Draw.schedule(() => {
      //    if (drumIdx < segments.length){
      //       segments[drumIdx].update({color:randomColor(0.25)});
      //    }
      // }, time);
   }

   function renderPattern(regenerating = false) {
      //       let seqEl = document.querySelector('.sequencer .steps');
      //       while (stepEls.length > state.pattern.length) {
      //          let { stepEl, gutterEl } = stepEls.pop();
      //          stepEl.remove();
      //          if (gutterEl) gutterEl.remove();
      //       }
      //       for (let stepIdx = 0; stepIdx < state.pattern.length; stepIdx++) {
      //          let step = state.pattern[stepIdx];
      //          let stepEl, gutterEl, cellEls;
      //          if (stepEls[stepIdx]) {
      //             stepEl = stepEls[stepIdx].stepEl;
      //             gutterEl = stepEls[stepIdx].gutterEl;
      //             cellEls = stepEls[stepIdx].cellEls;
      //          } else {
      //             stepEl = document.createElement('div');
      //             stepEl.classList.add('step');
      //             stepEl.dataset.stepIdx = stepIdx;
      //             seqEl.appendChild(stepEl);
      //             cellEls = [];
      //          }

      //          stepEl.style.flex = stepIdx % 2 === 0 ? state.swing : 1 - state.swing;

      //          if (!gutterEl && stepIdx < state.pattern.length - 1) {
      //             gutterEl = document.createElement('div');
      //             gutterEl.classList.add('gutter');
      //             seqEl.insertBefore(gutterEl, stepEl.nextSibling);
      //          } else if (gutterEl && stepIdx >= state.pattern.length) {
      //             gutterEl.remove();
      //             gutterEl = null;
      //          }

      //          if (gutterEl && stepIdx === state.seedLength - 1) {
      //             gutterEl.classList.add('seed-marker');
      //          } else if (gutterEl) {
      //             gutterEl.classList.remove('seed-marker');
      //          }

      //          for (let cellIdx = 0; cellIdx < DRUM_CLASSES.length; cellIdx++) {
      //             let cellEl;
      //             if (cellEls[cellIdx]) {
      //                cellEl = cellEls[cellIdx];
      //             } else {
      //                cellEl = document.createElement('div');
      //                cellEl.classList.add('cell');
      //                cellEl.classList.add(_.kebabCase(DRUM_CLASSES[cellIdx]));
      //                cellEl.dataset.stepIdx = stepIdx;
      //                cellEl.dataset.cellIdx = cellIdx;
      //                stepEl.appendChild(cellEl);
      //                cellEls[cellIdx] = cellEl;
      //             }
      //             if (step.indexOf(cellIdx) >= 0) {
      //                cellEl.classList.add('on');
      //             } else {
      //                cellEl.classList.remove('on');
      //             }
      //          }
      //          stepEls[stepIdx] = { stepEl, gutterEl, cellEls };

      //          let stagger = stepIdx * (300 / (state.patternLength - state.seedLength));
      //          setTimeout(() => {
      //             if (stepIdx < state.seedLength) {
      //                stepEl.classList.add('seed');
      //             } else {
      //                stepEl.classList.remove('seed');
      //                if (regenerating) {
      //                   stepEl.classList.add('regenerating');
      //                } else {
      //                   stepEl.classList.remove('regenerating');
      //                }
      //             }
      //          }, stagger);
      //       }

      //       setTimeout(repositionRegenerateButton, 0);
   }



   function regenerate(seed) {
      renderPattern(true);
      return generatePattern(seed, state.patternLength - seed.length).then(
         result => {
            state.pattern = result;
         }
      );
   }

   function updatePattern() {
      sequence.removeAll();
      state.pattern.forEach(function(drums, stepIdx) {
         sequence.at(stepIdx, {stepIdx:stepIdx, drums:drums});
      });
      renderPattern();
   }

   function toNoteSequence(pattern) {
      return mm.sequences.quantizeNoteSequence(
         {
            ticksPerQuarter: 220,
            totalTime: pattern.length / 2,
            timeSignatures: [
               {
                  time: 0,
                  numerator: 4,
                  denominator: 4
               }
            ],
            tempos: [
               {
                  time: 0,
                  qpm: 120
               }
            ],
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
      for (let { pitch, quantizedStartStep } of seq.notes) {
         res[quantizedStartStep].push(reverseMidiMapping.get(pitch));
      }
      return res;
   }
});

