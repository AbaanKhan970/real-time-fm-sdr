#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

"""
For the project release, we recommend working in 64-bit double format, as the
memory overhead is acceptable. Therefore, the normalization of 8-bit unsigned
I/Q samples has been performed in 64-bit double format.

By default, the block size must match the amount of I/Q data acquired in 40 ms
"""

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import math

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import fmDemodArctan, fmPlotPSD, my_Demod, my_own_coeff, my_own_coeffbpf, my_own_convolution, my_own_convolution_ns, block_delay
from fmPll import fmPll, fmPll_state, plotTime
# for take-home add your functions

# flag for mode 0 (0) vs mode 1 (1) 
mode = 0

if mode == 0:
	# the radio-frequency (RF) sampling rate
	rf_Fs = 2.4e6

	# the cutoff frequency to extract the FM channel from raw IQ data
	rf_Fc = 100e3

	# the number of taps for the low-pass filter
	rf_taps = 101

	# decimation rate for reducing sampling rate at the intermediate frequency (IF)
	rf_decim = 10

	# audio sampling rate (48 KSamples/sec)
	audio_Fs = 48e3

	# placeholders for audio channel settings
	audio_Fc = 16e3
	audio_decim = 5
	audio_taps = 101

elif mode == 1:
	# the radio-frequency (RF) sampling rate
	rf_Fs = 2.88e6

	# the cutoff frequency to extract the FM channel from raw IQ data
	rf_Fc = 100e3

	# the number of taps for the low-pass filter
	rf_taps = 101

	# decimation rate for reducing sampling rate at the intermediate frequency (IF)
	rf_decim = 8

	# audio sampling rate (48 KSamples/sec)
	audio_Fs = 36e3

	# placeholders for audio channel settings
	audio_exp = 1
	audio_Fc = 16e3	
	audio_decim = 10
	audio_taps = 101 * audio_exp
 
elif mode == 2:
    # the radio-frequency (RF) sampling rate
	rf_Fs = 2.4e6

	# the cutoff frequency to extract the FM channel from raw IQ data
	rf_Fc = 100e3

	# the number of taps for the low-pass filter
	rf_taps = 101

	# decimation rate for reducing sampling rate at the intermediate frequency (IF)
	rf_decim = 10

	# audio sampling rate (48 KSamples/sec)
	audio_Fs = 44.1e3

	# placeholders for audio channel settings
	audio_exp = 147
	audio_Fc = 16e3	
	audio_decim = 800
	audio_taps = 101 * audio_exp
 
elif mode == 3:
    # the radio-frequency (RF) sampling rate
	rf_Fs = 9.6e5

	# the cutoff frequency to extract the FM channel from raw IQ data
	rf_Fc = 100e3

	# the number of taps for the low-pass filter
	rf_taps = 101

	# decimation rate for reducing sampling rate at the intermediate frequency (IF)
	rf_decim = 5

	# audio sampling rate (48 KSamples/sec)
	audio_Fs = 44.1e3

	# placeholders for audio channel settings
	audio_exp = 147
	audio_Fc = 16e3	
	audio_decim = 640
	audio_taps = 101 * audio_exp


if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	#in_fname = "../data/iq_samples.raw"
	in_fname = "../data/samples0.raw"
	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")
	'''
	# IQ data is normalized between -1 and +1 in 32-bit float format
	iq_data = (np.float32(raw_data) - 128.0) / 128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")
	'''

	# IQ data is normalized between -1 and +1 in 64-bit double format
	iq_data = (np.float64(raw_data) - 128.0) / 128.0
	print("Reformatted raw RF data to 64-bit double format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for the front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc / (rf_Fs / 2), window=('hann'))

	# coefficients for the filter to extract mono audio
	'''
	if il_vs_th == 0:
		# to be updated by you during the in-lab session based on firwin
		# same principle as for rf_coeff (but different arguments, of course)
		audio_coeff = np.array([])
	else:
		# to be updated by you for the take-home exercise
		# with your own code for impulse response generation
		audio_coeff = np.array([])
	'''

	Fs_calculated = rf_Fs / rf_decim
	audio_coeff = my_own_coeff(Fs_calculated, audio_Fc, audio_taps)

	# set up the subfigures for plotting
	subfig_height = np.array([2, 2, 2]) # relative heights of the subfigures
	plt.rc('figure', figsize=(7.5, 7.5))	# the size of the entire figure
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)

	# change the block_size below as needed (check your custom constraints file)
	#
	# for real-time execution, by default, you must work with block sizes
	# that store 40 ms of data acquired in real-time
	# (you need to recalculate based on your custom rf_Fs)
	#
	#block_size = 1024 * rf_decim * audio_decim * 2
	block_size = int(rf_Fs * 40 / 1000)
	print(block_size)
	block_count = 0

	# states needed for continuity in block processing
	state_i_lpf_100k = np.zeros(rf_taps - 1)
	state_q_lpf_100k = np.zeros(rf_taps - 1)
	state_phase = 0
	# add state as needed for the mono channel filter

	state_audio_lpf_16k = np.zeros(rf_taps-1)
	prev_I = 0.0
	prev_Q = 0.0

	# audio buffer that stores all the audio blocks
	fm_demod_full = np.array([])
	audio_data = np.array([]) # used to concatenate filtered blocks (audio data)
	#stereo_left_data = np.array([])
	#stereo_right_data = np.array([])

	state_scr = np.zeros(rf_taps - 1)
	state_sce = np.zeros(rf_taps - 1)
	state_mixed = np.zeros(rf_taps - 1)

	state_pll = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0])

	mixed_data_full = np.array([])
	left_audio_full = np.array([])
	right_audio_full= np.array([])

	ncoOut_full = np.array([])

	# if the number of samples in the last block is less than the block size
	# it is fine to ignore the last few samples from the raw IQ file
	#while (block_count + 1) * block_size < len(iq_data):
	while (block_count + 1) * block_size < len(iq_data):

		# if you wish to have shorter runtimes while troubleshooting
		# you can control the above loop exit condition as you see fit
		print('Processing block ' + str(block_count))

		# filter to extract the FM channel (I samples are even, Q samples are odd)
		i_filt, state_i_lpf_100k = my_own_convolution(
				iq_data[block_count * block_size:(block_count + 1) * block_size:2], rf_coeff,  
				state_i_lpf_100k, rf_decim)
		q_filt, state_q_lpf_100k = my_own_convolution(\
				iq_data[block_count * block_size + 1:(block_count + 1) * block_size:2], rf_coeff,  
				state_q_lpf_100k, rf_decim)

		# downsample the I/Q data from the FM channel
		i_ds = i_filt[::rf_decim]
		q_ds = q_filt[::rf_decim]

		# FM demodulator
		'''
		if il_vs_th == 0:
			# already given to you for the in-lab
			# take particular notice of the "special" state-saving
			fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)
		else:
			# you will need to implement your own FM demodulation based on:
			# https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
			# see more comments on fmSupportLib.py - take particular notice that
			# you MUST have also "custom" state-saving for your own FM demodulator
			dummy_fm, dummy_state = np.array([]), np.array([])
		'''

		
		fm_demod, prev_I, prev_Q = my_Demod(i_ds, q_ds, prev_I, prev_Q)
		fm_demod_full = np.concatenate((fm_demod_full, fm_demod))

		if (mode == 0 or mode == 1):
			audio_filt, state_audio_lpf_16k = my_own_convolution(fm_demod, audio_coeff, state_audio_lpf_16k, audio_decim)

			# extract the mono audio data through filtering
			# downsample audio data
			audio_block = audio_filt[::audio_decim]

			# concatenate the most recently processed audio_block
			# to the previous blocks stored already in audio_data
			#
			audio_data = np.concatenate((audio_data, audio_block))
			#

			# to save runtime, select the range of blocks to log data
			# this includes both saving binary files and plotting PSD
			if block_count >= 10 and block_count < 12:

				# plot PSD of selected block after FM demodulation
				# (for easier visualization purposes we divide Fs by 1e3 to imply the kHz units on the x-axis)
				# (this scales the y axis of the PSD, but not the relative strength of different frequencies)
				ax0.clear()
				fmPlotPSD(ax0, fm_demod, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
						'Demodulated FM (block ' + str(block_count) + ')')
				# output binary file name (where samples are written from Python)
				fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
				'''
				# create binary file where each sample is a 32-bit float
				fm_demod.astype('float32').tofile(fm_demod_fname)
				'''

				# create binary file where each sample is a 64-bit double
				fm_demod.astype('float64').tofile(fm_demod_fname)

				# save figure to file
				fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png")

			block_count += 1
   
		elif (mode == 2 or mode == 3):
			# upsampling audio data
			upsampled = []
			fm_demod = np.array(fm_demod)
			upsampled = np.zeros(len(fm_demod) * audio_exp)
			upsampled[::audio_exp] = fm_demod
			audio_filt, state_audio_lpf_16k = my_own_convolution(upsampled, audio_coeff, state_audio_lpf_16k, audio_decim)

			# extract the mono audio data through filtering
			# downsample audio data
			audio_block = audio_filt[::audio_decim]

			# concatenate the most recently processed audio_block
			# to the previcdous blocks stored already in audio_data
			#
			audio_data = np.concatenate((audio_data, audio_block))
			block_count += 1
		

			block_count += 1
	
	print('Finished processing all the blocks from the recorded I/Q samples')

	# write mono data to file
	out_fname = "../data/fmMonoBlock0.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")
	# uncomment assuming you wish to show some plots
	# plt.show()


	
