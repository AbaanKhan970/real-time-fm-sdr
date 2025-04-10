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
channel = 1 #mono: 0, stereo: 1, rds: 2
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


if __name__ == "__main__":

	# read the raw IQ data from the recorded file
	# IQ data is assumed to be in 8-bits unsigned (and interleaved)
	#in_fname = "../data/iq_samples.raw"
	in_fname = "../data/iq_samples2.raw"
	#in_fname = "../data/iq_sample3.raw"
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
	rf_coeff = my_own_coeff(rf_Fs, rf_Fc, rf_taps)

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

		if (mode == 0):
			#STEREO
			Flow_scr = 18.5e3
			Fhigh_scr = 19.5e3

			Flow_sce = 22e3
			Fhigh_sce = 54e3

			#STEREO CARRIER RECOVERY
			#bpf
			scr_bpf_coeff = my_own_coeffbpf(Flow_scr, Fhigh_scr, Fs_calculated, rf_taps)
			scr_filt, state_scr = my_own_convolution(fm_demod, scr_bpf_coeff, state_scr, 1)
			print('STEREO CARRIER RECOVERY BPF DONE')
			#pll + nco
			ncoOut, state_pll = fmPll_state(scr_filt, 19e3, Fs_calculated, state_pll)
			print('STEREO CARRIER RECOVERY PLL + NCO DONE')	

			ncoOut_full = np.concatenate((ncoOut_full, ncoOut))					

			#STEREO CHANNEL EXTRACTION
			sce_bpf_coeff = my_own_coeffbpf(Flow_sce, Fhigh_sce, Fs_calculated, rf_taps)
			sce_filt, state_sce = my_own_convolution(fm_demod, sce_bpf_coeff, state_sce, 1)
			print('STEREO CHANNEL EXTRACTION BPF DONE')
			#MIXERdecim
			mixed = np.array([0.0] * len(fm_demod))
			for i in range(len(fm_demod)):
				mixed[i] = sce_filt[i] * ncoOut[i]

			# mixed = np.multiply(ncoOut[0:len(sce_filt):1], sce_filt)

			print('MIXING DONE')
			print('len mixed: ', len(mixed))

			#LPF same as mono
			stereo_coeff = my_own_coeff(Fs_calculated, audio_Fc, audio_taps)
			mixed_filt, state_mixed = my_own_convolution(mixed, stereo_coeff, state_mixed, audio_decim)
			print('LPF DONE')

			for i in range(len(mixed_filt)):
				mixed_filt[i] *= 2

			
			# downsample same as mono
			mixed_data = mixed_filt[::audio_decim]

			mixed_data_full = np.concatenate((mixed_data_full, mixed_data))
			print(len(mixed_data))
			print('len mixed_data: ', len(mixed_data_full))
			print('DOWNSAMPLING DONE')
			
            

			delay_block = np.array([0.0]*((audio_taps - 1) // 2))
			delayed_audio_block, delay_block = block_delay(fm_demod, delay_block)
            
			audio_filt, state_audio_lpf_16k = my_own_convolution(delayed_audio_block, audio_coeff, state_audio_lpf_16k, audio_decim)

			# extract the mono audio data through filtering
			# downsample audio data
			audio_block = audio_filt[::audio_decim]

			# concatenate the most recently processed audio_block
			# to the previous blocks stored already in audio_data
			#
			audio_data = np.concatenate((audio_data, audio_block))
			print('len audio block: ', len(audio_block))
			#
			'''
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
				

				# create binary file where each sample is a 32-bit float
				fm_demod.astype('float32').tofile(fm_demod_fname)
				

				# create binary file where each sample is a 64-bit double
				fm_demod.astype('float64').tofile(fm_demod_fname)

				# save figure to file
				fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png")
			'''
		


			
			left_audio = np.zeros(len(mixed_data))
			right_audio = np.zeros(len(mixed_data))

			#delay block (all-pass filter)
			#delay_block = int((audio_taps - 1) / 2)
			#delay_block = np.array([0.0]*((audio_taps - 1) // 2))
			#if block_count == 0:			
			#	delayed_audio_block = np.concatenate((np.zeros(delay_coeff), audio_block[:-delay_coeff]))
			#else:
			#	delayed_audio_block = np.concatenate((audio_data[-delay_coeff:], audio_block[:-delay_coeff]))

			#delayed_audio_block, delay_block = block_delay(audio_block, delay_block)
			
			print('len delayed audio block: ', len(delayed_audio_block))

			#STEREO COMBINER
			#for i in range(len(mixed_data)):
			#	left_audio[i] = (mixed_data[i] + delayed_audio_block[i])
			#	right_audio[i] = (mixed_data[i] - delayed_audio_block[i])
			
			for i in range(len(mixed_data)):
				left_audio[i] = (audio_block[i] + mixed_data[i]) 
				right_audio[i] = (audio_block[i] - mixed_data[i])
			
			left_audio_full = np.concatenate((left_audio_full, left_audio))
			right_audio_full = np.concatenate((right_audio_full, right_audio))

			print('len left audio full: ', len(left_audio_full))
			print('len right audio full: ', len(right_audio_full))

			print('COMBINING DONE')

			# if block_count >= 10 and block_count < 12:

				# plot PSD of selected block after FM demodulation
				# (for easier visualization purposes we divide Fs by 1e3 to imply the kHz units on the x-axis)
				# (this scales the y axis of the PSD, but not the relative strength of different frequencies)
				# ax0.clear()
				# fmPlotPSD(ax0, fm_demod, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
				# 		'Demodulated FM (block ' + str(block_count) + ')')
				# # output binary file name (where samples are written from Python)
				# #fm_demod_fname = "../data/fm_demod_" + str(block_count) + ".bin"
				

				# # create binary file where each sample is a 32-bit float
				# #fm_demod.astype('float32').tofile(fm_demod_fname)
				

				# # create binary file where each sample is a 64-bit double
				# #fm_demod.astype('float64').tofile(fm_demod_fname)
				# fmPlotPSD(ax1, left_audio, audio_Fs / 1e3, subfig_height[1], \
				# 		'Left Audio(block ' + str(block_count) + ')')

				# fmPlotPSD(ax2, right_audio, audio_Fs / 1e3, subfig_height[2], \
				# 		'Right Audio (block ' + str(block_count) + ')')
				# # save figure to file
				# fig.savefig("../data/fmMonoBlock" + str(block_count) + ".png") 

		block_count += 1
	
	time = np.arange(len(ncoOut)) / Fs_calculated
	#plotTime(ncoOut, time)
	print('Finished processing all the blocks from the recorded I/Q samples')

	# write mono data to file
	out_fname = "../data/fmMonoBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write stereo data to file
	out_fname = "../data/fmStereoLeftBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((left_audio_full / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write audio data to file
	out_fname = "../data/fmStereoRightBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((right_audio_full / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write audio data to file
	out_fname = "../data/fmStereoMixedBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((mixed_data_full / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# uncomment assuming you wish to show some plots
	# plt.show()


	