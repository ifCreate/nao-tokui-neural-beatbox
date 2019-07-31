/* Variable declaration statement. All the parameters that can be
adjusted on the right side can be adjusted in the following statement. */

// The value range of the parameter newPatternLength ranges from 1 to 32,
// and preferably be a multiple of 4. The value range of the parameter
// newTempo ranges from 20 to 240, and preferably be an integer.
// The value range of the parameter newTempo ranges from 0.5 to 0.7.
// The value range of the parameter newTemperature ranges from 0 to
// positive infinity, and must be a positive.
var newPatternLength = 16;
var newTempo = 140;
var newSwing = 0.6;
var newTemperature = 1.5;
var newSeed = [[0],[1,2],[5],[8]];

/* Reset the contents of the above variables, the definition of the
setting method is at the backend, the code is not shown here. */
setPatternLength(newPatternLength);
setTempo(newTempo);
setSwing(newSwing);
setTemperature(newTemperature);
setSeed(newSeed);


/*This part is the sound prediction module. Press F12 to view the
results in the console.*/
$.ajax({
	url: 'demo/data.json',
	success: function(data) {
		var train_total_num = 1;
		var melCount = 128;
		var specLength = 32;
		// Create tf.tensor2d
		// This audio classification model expects spectrograms of [128, 32]
		// (# of melbanks: 128 / duration: 32 FFT windows)
		const tfbuffer1 = tf.buffer([train_total_num, melCount, specLength]);

		// Initialize the tfbuffer.
		for (let a = 0; a < train_total_num; a++) {
			for (let i = 0; i < melCount; i++) {
				for (let j = 0; j < specLength; j++) {
					tfbuffer1.set(MIN_DB, a, i, j);
				}
			}
		}
		for (let a = 0; a < train_total_num; a++) {
			for (let i = 0; i < melCount; i++) {
				for (let j = 0; j < specLength; j++) {
					// cantion: needs to transpose the matrix
					tfbuffer1.set(data['train_melspecs'][a][j][i], a, i, j);
				}
			}
		}

		// Reshape for prediction
		train_tensor = tfbuffer1.toTensor(); // tf.buffer -> tf.tensor
		// [null, 128, 128, 1]
		train_tensor = tf.reshape(train_tensor, [train_tensor.shape[0],
			train_tensor.shape[1], train_tensor.shape[2], 1]);

		let predictions = tfmodel.predict(train_tensor);
		predictions = predictions.flatten().dataSync();
		//change the next line to 'alert(predictions)'
		// then you can check the result in alerts
		console.log(predictions);
	}
});

