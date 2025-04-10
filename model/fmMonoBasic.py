#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Copyright by Nicola Nicolici
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

"""
Ten prerecorded I/Q sample files can be downloaded from the link sent via email
and posted on Avenue.

There is NO need to copy all the sample files at once into your data sub-folder.
Instead, it is recommended to download one file at a time and overwrite the
iq_samples.raw file in the data sub-folder, which is the default path used by the script.

For those of you with the RF dongle and Raspberry Pi kit at home, the command-line
instructions for recording RF data are provided below. After installing the necessary
drivers to work with the RF dongle, the 8-bit unsigned values for the I/Q pairs can be
recorded using the following command:

rtl_sdr -f 99.9M -s 2.4M - > iq_samples.raw

The above command assumes that we are tuned to the FM station at 99.9 MHz, using an
RF sample rate of 2.4 Msamples/sec, and saving the data to a file named iq_samples.raw
(you can change the filename as needed).

For the above command, data acquisition runs indefinitely and must be stopped manually
by pressing Ctrl+C. If you want to stop recording after a predefined number of samples,
e.g., 12 million I/Q pairs (equivalent to 5 seconds at 2.4 Msamples/sec), you can
include an additional argument:

rtl_sdr -f 99.9M -s 2.4M -n 12000000 - > iq_samples.raw

To verify that the raw I/Q data has been recorded properly, place the file in your
project repository's data sub-folder and run the Python script from the model sub-folder.
The script should generate both .png image files (showing PSD estimates) and a .wav file.

In the source code below (at the beginning of the main function), you can observe where
the raw_data is read and where the normalization of the 8-bit unsigned I/Q samples to
32-bit float samples (in the range -1 to +1) is performed. While using 32-bit floats and
normalizing to the range -1 to +1 are optional choices (commonly adopted by many
third-party SDR software implementations), it is up to each project group to decide how
to handle the 8-bit unsigned I/Q samples in their Python model and C++ implementation.
For example, one can choose how to reformat the data in the range -1 to +1 in 64-bit double
format (as shown in the lines commented below the normalization to 32-bit floats). Note,
however, it is recommended to use the same data format in both the Python model and C++.

A final but critical note: the .gitignore file should NOT be modified to allow pushing
.raw files to GitHub. Including .raw files in the repository will result in very large
repositories that take excessive time to clone, pull, or push. As outlined in the
reference .gitignore file, only source files should be kept in your repositories.
"""

"""
For the project release, we recommend working in 64-bit double format, as the
memory overhead is acceptable. Therefore, the normalization of 8-bit unsigned
I/Q samples has been performed in 64-bit double format.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
import math

# use fmDemodArctan and fmPlotPSD
from fmSupportLib import fmDemodArctan, fmPlotPSD, my_Demod, my_own_coeff, my_own_coeffbpf, my_own_convolution, my_own_convolution_ns
from fmPll import fmPll
# for take-home add your functions

# flag for mode 0 (0) vs mode 1 (1) 
mode = 0
channel = 1

if mode == 0:
	# the radio-frequency (RF) sampling rate
	rf_Fs = 2.4e6

	# the cutoff frequency to extract the FM channel from raw IQ data
	rf_Fc = 100e3

	# the number of taps for the low-pass filter
	rf_taps = 151

	# decimation rate for reducing sampling rate at the intermediate frequency (IF)
	rf_decim = 10

	# audio sampling rate (48 KSamples/sec)
	audio_Fs = 48e3

	# placeholders for audio channel settings
	audio_Fc = 16e3
	audio_decim = 5
	audio_taps = 151

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

	# read the raw IQ data
	in_fname = "../data/iq_samples2.raw"
	raw_data = np.fromfile(in_fname, dtype='uint8')
	print("Read raw RF data from \"" + in_fname + "\" in unsigned 8-bit format")

	'''
	# normalize raw IQ data to 32-bit float format (-1 to +1)
	iq_data = (np.float32(raw_data) - 128.0) / 128.0
	print("Reformatted raw RF data to 32-bit float format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")
	'''

	# IQ data is normalized between -1 and +1 in 64-bit double format
	iq_data = (np.float64(raw_data) - 128.0) / 128.0
	print("Reformatted raw RF data to 64-bit double format (" + str(iq_data.size * iq_data.itemsize) + " bytes)")

	# coefficients for front-end low-pass filter
	rf_coeff = signal.firwin(rf_taps, rf_Fc / (rf_Fs / 2), window=('hann'))

	# filter to extract the FM channel
	i_filt = signal.lfilter(rf_coeff, 1.0, iq_data[0::2])
	q_filt = signal.lfilter(rf_coeff, 1.0, iq_data[1::2])
	print("IQ Filtering Done")

	# downsample the FM channel
	i_ds = i_filt[::rf_decim]
	q_ds = q_filt[::rf_decim]
	print("IQ Downsampling Done")
	
	
	# set up subfigures for plotting
	
	subfig_height = np.array([2, 2, 2])
	plt.rc('figure', figsize=(7.5, 7.5))
	fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
	fig.subplots_adjust(hspace = .6)
    
	# FM demodulator
	fm_demod, dummy = fmDemodArctan(i_ds, q_ds)
	print("Demodulation Done")
	# PSD after FM demodulation
	# (for easier visualization purposes we divide Fs by 1e3 to imply the kHz units on the x-axis)
	# (this scales the y axis of the PSD, but not the relative strength of different frequencies)
	#fmPlotPSD(ax0, fm_demod, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
	#	'Demodulated FM (full recording)')
	
	# placeholder for audio filter coefficients
	Fs_calculated = rf_Fs / rf_decim
	audio_coeff = my_own_coeff(Fs_calculated, audio_Fc, audio_taps)
	
	
	# extract mono audio data (placeholder)
	#audio_filt = my_own_convolution_ns(audio_coeff, fm_demod)
	audio_filt = signal.lfilter(audio_coeff, 1.0, fm_demod)
	print("Mono Filtering Done")
	# PSD for mono audio (placeholder)
	#fmPlotPSD(ax1, audio_filt, (rf_Fs / rf_decim) / 1e3, subfig_height[1], 'Extracted Mono')

	# downsample mono audio (placeholder)
	audio_data = audio_filt[::audio_decim]
	print("Mono Downsampling Done")
	# PSD for downsampled audio (placeholder)
	#fmPlotPSD(ax2, audio_data, audio_Fs / 1e3, subfig_height[2], 'Mono Audio')

	# save plots
	#fig.savefig("../data/fmMonoLR.png")
	#plt.show()
	
	if(channel == 1):
		Flow_scr = 18.5e3
		Fhigh_scr = 19.5e3

		Flow_sce = 22e3
		Fhigh_sce = 54e3

		#STEREO CARRIER RECOVERY
		#bpf
		scr_bpf_coeff = my_own_coeffbpf(Flow_scr, Fhigh_scr, Fs_calculated, audio_taps)
		#scr_filt = my_own_convolution_ns(scr_bpf_coeff, fm_demod)
		scr_filt = signal.lfilter(scr_bpf_coeff, 1.0, fm_demod)
		
		print('STEREO CARRIER RECOVERY BPF DONE')
		print(len(scr_filt))
		#fmPlotPSD(ax1, scr_filt, (rf_Fs / rf_decim) / 1e3, subfig_height[1], \
		#'Stereo Carrier Band Pass Filtered')



		#pll + nco
		ncoOut = fmPll(scr_filt, 19e3, Fs_calculated, 2)
		
		print('STEREO CARRIER RECOVERY PLL + NCO DONE')		
		print(len(ncoOut))
		#fmPlotPSD(ax0, ncoOut, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
		#'Stereo Carrier')	


		

		#STEREO CHANNEL EXTRACTION
		sce_bpf_coeff = my_own_coeffbpf(Flow_sce, Fhigh_sce, Fs_calculated, audio_taps)
		#sce_filt = my_own_convolution_ns(sce_bpf_coeff, fm_demod)
		sce_filt = signal.lfilter(sce_bpf_coeff, 1.0, fm_demod)

		print('STEREO CHANNEL EXTRACTION BPF DONE')
		print(len(sce_filt))
		#fmPlotPSD(ax1, sce_filt, (rf_Fs / rf_decim) / 1e3, subfig_height[1], \
		#'Stereo Extracted, Band-Pass Filtered')	
	
		
				
		
		#MIXER
		mixed = np.multiply(ncoOut[0:len(sce_filt):1], sce_filt)

		#mixed = np.zeros(len(fm_demod))
		#for i in range(len(fm_demod)):
		#	mixed[i] = sce_filt[i] * ncoOut[i] 
			
		print('MIXING DONE')
		print(len(mixed))
		#fmPlotPSD(ax0, mixed, (rf_Fs / rf_decim) / 1e3, subfig_height[0], \
		#'Stereo extracted x carrier')		
		
		

		#LPF same as mono
		#mixed_filt = my_own_convolution_ns(audio_coeff, mixed)
		mixed_filt = signal.lfilter(audio_coeff, 1.0, mixed)



		recovered = np.multiply(mixed_filt, 2)
		#for i in range(len(mixed_filt)):
		#	mixed_filt[i] *= 2
			
		print(len(recovered))
		#fmPlotPSD(ax1, recovered, (rf_Fs / rf_decim) / 1e3, subfig_height[1], \
		#'Mixed Signal Filtered')	
		
		
		print('LPF DONE')
		# downsample same as mono
		mixed_data = recovered[::audio_decim]
		print('DOWNSAMPLING DONE')
		print(len(mixed_data))
		
		#fmPlotPSD(ax1, mixed_data, audio_Fs / 1e3, subfig_height[1], \
		#'Mixed Audio')

		
		left_audio = np.zeros(len(audio_data))
		right_audio = np.zeros(len(audio_data))


		#delay block (all-pass filter)
		delay_coeff = int((audio_taps - 1) / 2)

		zeroes = np.zeros(delay_coeff)
		
		delayed_audio_block = audio_data[:-delay_coeff]				
		delayed_audio_block = np.concatenate((zeroes, delayed_audio_block))


		#STEREO COMBINER
		for i in range(len(audio_data)):
			left_audio[i] = (audio_data[i] + mixed_data[i]) / 2
			right_audio[i] = (audio_data[i] - mixed_data[i]) / 2
		print('COMBINING DONE')	

		#fmPlotPSD(ax2, left_audio, audio_Fs / 1e3, subfig_height[2], \
		#'Left Audio')

		#fmPlotPSD(ax2, right_audio, audio_Fs / 1e3, subfig_height[2], \
		#'Right Audio')

		#fig.savefig("../data/fmStereoRight.png")
		#plt.show()
		
	print('Finished processing all the blocks from the recorded I/Q samples')


	# write audio data to file
	out_fname = "../data/fmMonoBasic.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((audio_data / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write stereo data to file
	out_fname = "../data/fmStereoLeftBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((left_audio / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write audio data to file
	out_fname = "../data/fmStereoRightBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((right_audio / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")

	# write audio data to file
	out_fname = "../data/fmStereoMixedBlock.wav"
	wavfile.write(out_fname, int(audio_Fs), np.int16((mixed_data / 2) * 32767))
	print("Written audio samples to \"" + out_fname + "\" in signed 16-bit format")
