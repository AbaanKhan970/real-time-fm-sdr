#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import numpy as np
import math, cmath

import matplotlib.pyplot as plt
import numpy as np


#
# you should use the demodulator based on arctan given below as a reference
#
# in order to implement your OWN FM demodulator without the arctan function,
# a very good and to-the-point description is given by Richard Lyons at:
#
# https://www.embedded.com/dsp-tricks-frequency-demodulation-algorithms/
#
# the demodulator boils down to implementing equation (13-117) from above, where
# the derivatives are nothing else but differences between consecutive samples
#
# needless to say, you should not jump directly to equation (13-117)
# rather try first to understand the entire thought process based on calculus
# identities, like derivative of the arctan function or derivatives of ratios
#

# use the four quadrant arctan function for phase detect between a pair of
# IQ samples; then unwrap the phase and take its derivative to FM demodulate

def my_own_coeff(Fs, Fc, N_taps):
	Norm_c = Fc/(Fs/2)
	Middle_i = (N_taps - 1)/2

	h = np.zeros(N_taps)


	for i in range(N_taps):
		if i == (N_taps - 1)/2 :
			h[i] = Norm_c
		else:
			h[i] = Norm_c * np.sinc(Norm_c*(i-Middle_i))
		h[i] = h[i] * (0.5 - (0.5 * np.cos((2 * np.pi * i)/(N_taps-1))))
	
	return h


def my_own_convolution_ns(h,x):
	
	y=np.zeros(len(h) + len(x) - 1)

	for n in range(len(y)):
		for k in range(len(h)):
			if(n-k>=0 and n-k<len(x)):
				y[n] += h[k] * x[n-k]
	
	return y


def my_own_convolution(x, h, state, decim):
	#print(state)
	y = np.zeros(len(x))
	for n in range(0,len(y), decim):
		for k in range(len(h)):
			if n-k >= 0:
				y[n] += h[k] * x[n - k]
			else:
				# use state from previous block
				#print(len(state)+n-k)
				y[n] += h[k] * state[len(state) + n - k]
	#save state for next block
	state = x[-(len(h) - 1):]
	return y, state
'''
def my_own_convolution(x, h, state, decim):
    y = np.zeros(len(x))
    for n in range(0, len(y), decim):
        for k in range(len(h)):
            if n - k >= 0:
                y[n] += h[k] * x[n - k]
            else:
                # Handle upsampled state properly
                state_idx = len(state) + n - k
                if state_idx >= 0 and state_idx < len(state):
                    y[n] += h[k] * state[state_idx]
                else:
                    y[n] += 0  # Zero-pad if beyond state
    # Save state for next block - only keep the needed samples
    state = x[-(len(h) - 1):]
    return y, state

def my_own_convolution(x, h, state, decim):
    y = np.zeros(len(x) // decim)  # Only store output samples we'll keep
    h_len = len(h)
    state_len = len(state)
    
    for n in range(0, len(y)):
        output_idx = n * decim
        y[n] = 0
        
        # Current input samples
        for k in range(min(h_len, output_idx + 1)):
            y[n] += h[k] * x[output_idx - k]
        
        # State samples
        for k in range(output_idx + 1, h_len):
            state_idx = state_len + output_idx - k
            if 0 <= state_idx < state_len:
                y[n] += h[k] * state[state_idx]
    
    # Update state (only keep what we need for next block)
    new_state = np.zeros_like(state)
    if len(x) >= h_len - 1:
        new_state[:] = x[-(h_len - 1):]
    else:
        new_state[-(len(x)):] = x[:]
        new_state[:-(len(x))] = state[-(h_len - 1 - len(x)):]
    
    return y, new_state

'''
def resampler(x, h, state, decim, up):

	y = np.zeros(int(len(x) * (up / decim)))
	for n in range(0,len(y)):
		y[n] = 0
		phase = (n * decim) % up
		for k in range(phase, len(h), up):
			idx = (n * decim - k) / up
			if(idx >= 0):
				y[n] += h[k] * x[int(idx)]
			
			else:
				y[n] += h[k] * state[int(len(state) + idx)]
		
	state = x[-(len(h) - 1):]

	return y, state

def my_own_coeffbpf(Flow, Fhigh, Fs, N_taps):
	center = (N_taps-1) // 2
	Fmid = (Fhigh + Flow) / 2.0 

	scale_factor = 0.0
	h = np.array([0.0] * N_taps)

	for k in range(N_taps):
		n = k - center

		if (n == 0) :
			#h[k] = ((2.0*(Fhigh/ Fs)) - (2.0*(Flow/ Fs)))
			h[k] = 2.0*((Fhigh - Flow)/Fs)
		else:
			h[k] = (np.sin((2.0 * np.pi * n * Fhigh)/Fs) - np.sin((2.0 * np.pi * n * Flow)/Fs))/ (np.pi * n)

		h[k] = h[k] * float(0.5 - (0.5 * np.cos((2.0 * np.pi * k)/(N_taps-1))))
		scale_factor += h[k] * float(np.cos((2.0 * np.pi * n * Fmid)/Fs))

	for k in range(N_taps):
		h[k] = float(h[k] / scale_factor)
	
 	
	return h


def block_delay(x, delay_state):
	output = np.concatenate((delay_state, x[:-len(delay_state)]))
	delay_state = x[-len(delay_state):]
	return output, delay_state


def fmDemodArctan(I, Q, prev_phase = 0.0):

	# the default prev_phase phase is assumed to be zero, however
	# take note in block processing it must be explicitly controlled

	# empty vector to store the demodulated samples
	fm_demod = np.empty(len(I))

	# iterate through each of the I and Q pairs
	for k in range(len(I)):

		# use the atan2 function (four quadrant version) to detect angle between
		# the imaginary part (quadrature Q) and the real part (in-phase I)
		current_phase = math.atan2(Q[k], I[k])

		# we need to unwrap the angle obtained in radians through arctan2
		# to deal with the case when the change between consecutive angles
		# is greater than Pi radians (unwrap brings it back between -Pi to Pi)
		[prev_phase, current_phase] = np.unwrap([prev_phase, current_phase])

		# take the derivative of the phase
		fm_demod[k] = current_phase - prev_phase

		# save the state of the current phase
		# to compute the next derivative
		prev_phase = current_phase

	# return both the demodulated samples as well as the last phase
	# (the last phase is needed to enable continuity for block processing)
	return fm_demod, prev_phase

def my_Demod(I, Q, prev_I = 0.0,  prev_Q = 0.0):
	fm_demod = np.empty(len(I))

	for k in range(len(I)):
		if((I[k]**2 + Q[k]**2) == 0):
			fm_demod[k] =  0
		else:
			numerator =  (I[k]*(Q[k]-prev_Q)) - (Q[k]*(I[k]-prev_I))
			denominator =  (I[k]**2 + Q[k]**2)
			fm_demod[k] = numerator/denominator

		prev_I = I[k]
		prev_Q = Q[k]

	return fm_demod, prev_I, prev_Q

# custom function for DFT that can be used by the PSD estimate
def DFT(x):

	# number of samples
	N = len(x)

	# frequency bins
	Xf = np.zeros(N, dtype='complex')

	# iterate through all frequency bins/samples
	for m in range(N):
		for k in range(N):
			Xf[m] += x[k] * cmath.exp(1j * 2 * math.pi * ((-k) * m) / N)

	# return the vector that holds the frequency bins
	return Xf

# custom function to estimate PSD based on the Bartlett method
# this is less accurate than the Welch method used in some packages
# however, as the visual inspections confirm, the estimate gives
# the user a "reasonably good" view of the power spectrum
def estimatePSD(samples, NFFT, Fs):

	# rename the NFFT argument (notation consistent with matplotlib.psd)
	# to freq_bins (i.e., frequency bins for which we compute the spectrum)
	freq_bins = NFFT
	# frequency increment (or resolution of the frequency bins)
	df = Fs / freq_bins

	# create the frequency vector to be used on the X axis
	# for plotting the PSD on the Y axis (only positive freq)
	freq = np.arange(0, Fs / 2, df)

	# design the Hann window used to smoothen the discrete data in order
	# to reduce the spectral leakage after the Fourier transform
	hann = np.empty(freq_bins)
	for i in range(len(hann)):
		hann[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (freq_bins - 1)))

	# create an empty list where the PSD for each segment is computed
	psd_list = []

	# samples should be a multiple of frequency bins, so
	# the number of segments used for estimation is an integer
	# note: for this to work you must provide an argument for the
	# number of frequency bins not greater than the number of samples!
	no_segments = int(math.floor(len(samples) / float(freq_bins)))

	# iterate through all the segments
	for k in range(no_segments):

		# apply the hann window (using pointwise multiplication)
		# before computing the Fourier transform on a segment
		windowed_samples = samples[k * freq_bins:(k + 1) * freq_bins] * hann

		# compute the Fourier transform using the built-in FFT from numpy
		Xf = np.fft.fft(windowed_samples, freq_bins)

		# since input is real, we keep only the positive half of the spectrum
		# however, we will also add the signal energy of negative frequencies
		# to have a better and more accurate PSD estimate when plotting
		Xf = Xf[0:int(freq_bins / 2)] # keep only positive freq bins
		psd_seg = (1 / (Fs * freq_bins / 2)) * (abs(Xf)**2) # compute signal power
		psd_seg = 2 * psd_seg # add the energy from the negative freq bins

		# append to the list where PSD for each segment is stored
		# in sequential order (first segment, followed by the second one, ...)
		psd_list.extend(psd_seg)

	# iterate through all the frequency bins (positive freq only)
	# from all segments and average them (one bin at a time ...)
	psd_seg = np.zeros(int(freq_bins / 2))
	for k in range(int(freq_bins / 2)):
		# iterate through all the segments
		for l in range(no_segments):
			psd_seg[k] += psd_list[k + l * int(freq_bins / 2)]
		# compute the estimate for each bin
		psd_seg[k] = psd_seg[k] / no_segments

	# translate to the decibel (dB) scale
	psd_est = np.zeros(int(freq_bins / 2))
	for k in range(int(freq_bins / 2)):
		psd_est[k] = 10 * math.log10(psd_seg[k])

	# the frequency vector and PSD estimate
	return freq, psd_est

# custom function to format the plotting of the PSD
def fmPlotPSD(ax, samples, Fs, height, title):

	# adjust grid lines as needed
	x_major_interval = (Fs / 12)
	x_minor_interval = (Fs / 12) / 4
	y_major_interval = 20
	x_epsilon = 1e-3
	# adjust x/y range as needed
	x_max = x_epsilon + Fs / 2
	x_min = 0
	y_max = 10
	y_min = y_max - 100 * height

	ax.psd(samples, NFFT=512, Fs=Fs)
	#
	# below is the custom PSD estimate, which is based on the Bartlett method
	# it is less accurate than the PSD from matplotlib, however it is sufficient
	# to help us visualize the power spectra on the acquired/filtered data
	#
	#freq, my_psd = estimatePSD(samples, NFFT=512, Fs=Fs)
	#ax.plot(freq, my_psd)
	#
	ax.set_xlim([x_min, x_max])
	ax.set_ylim([y_min, y_max])
	ax.set_xticks(np.arange(x_min, x_max, x_major_interval))
	ax.set_xticks(np.arange(x_min, x_max, x_minor_interval), minor=True)
	ax.set_yticks(np.arange(y_min, y_max, y_major_interval))
	ax.grid(which='major', alpha=0.75)
	ax.grid(which='minor', alpha=0.25)
	ax.set_xlabel('Frequency (kHz)')
	ax.set_ylabel('PSD (db/Hz)')
	ax.set_title(title)

##############################################################
# New code as part of benchmarking/testing and the project
##############################################################

# custom function to estimate PSD using the matrix approach
def matrixPSD(samples, NFFT, Fs):

	freq_bins = NFFT
	df = Fs / freq_bins
	freq = np.arange(0, Fs / 2, df)
	no_segments = int(math.floor(len(samples) / float(freq_bins)))

	# generate the DFT matrix for the given size N
	dft_matrix = np.empty((freq_bins, freq_bins), dtype='complex')
	for m in range(freq_bins):
		for k in range(freq_bins):
			dft_matrix[m, k] = cmath.exp(1j * 2 * math.pi * ((-k) * m) / freq_bins)

	# generate the Hann window for the given size N
	hann_window = np.empty(freq_bins, dtype='float')
	for i in range(freq_bins):
		hann_window[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (freq_bins - 1)))

	# apply Hann window and perform matrix multiplication using nested loops
	Xf = np.zeros((no_segments, freq_bins), dtype='complex')
	for seg in range(no_segments):
		for m in range(freq_bins):
			for k in range(freq_bins):
				Xf[seg][m] += samples[seg * freq_bins + k] * hann_window[k] * dft_matrix[m][k]

	# compute power, keep only positive frequencies, average across segments, and convert to dB
	psd_est = np.zeros(int(freq_bins / 2))  # same as (freq_bins // 2)
	for m in range(freq_bins // 2):
		sum_power = 0.0
		for seg in range(no_segments):
			sum_power += (1 / ((Fs / 2) * (freq_bins / 2))) * (abs(Xf[seg][m]) ** 2)
		psd_est[m] += 10 * math.log10(sum_power / no_segments)

	return freq, psd_est

# function to unit test PSD estimation
def psdUnitTest(min=-1, max=1, Fs=1e3, size=1024, NFFT=128):

	# generate random samples for testing
	samples = np.random.uniform(low=min, high=max, size=size)

	# calculate reference PSD
	freq_ref, psd_ref = estimatePSD(samples, NFFT, Fs)

	# calculate PSD using the matrix-based function
	freq_mat, psd_mat = matrixPSD(samples, NFFT, Fs)

	# check if all the values are close within the given tolerance
	if not np.allclose(freq_ref, freq_mat, atol=1e-4):
		print("Comparison between reference frequency vectors fails")

	if not np.allclose(psd_ref, psd_mat, atol=1e-4):
		print("Comparison between reference estimate PSD and matrix PSD fails")
		print("Reference PSD:", psd_ref)
		print("Matrix PSD   :", psd_mat)
		print("Maximum difference:", np.max(np.abs(psd_ref - psd_mat)))
	else:
		print(f"Unit test for matrix PSD transform passed.")

if __name__ == "__main__":

	
	# this unit test (when uncommented) will confirm that
	# estimate PSD and matrix PSD are equivalent to each other
	#psdUnitTest()
	

	# do nothing when this module is launched on its own
	pass




def plot_rrc_with_offsets(rrc_i, sps, offset):
    
    sample_numbers = np.arange(len(rrc_i))
    
    
    plt.figure(figsize=(12, 6))
    
    
    plt.plot(sample_numbers, rrc_i, 'b-', label='Signal after the root-raised cosine filter')
    
    
    offset_positions = np.arange(offset, len(rrc_i), sps)
    
    
    for pos in offset_positions:
        if pos < len(rrc_i):  
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.7)
            
            plt.plot([pos, pos], [0, rrc_i[pos]], 'r-', linewidth=1.5, alpha=0.7)
    
    
    plt.xlabel('Sample #')
    plt.ylabel('Signal strength')
    plt.title('Waveform before timing recovery with optimal sampling points')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    
    plt.xlim(0, len(rrc_i))
    
    plt.show()




#clock data recovery
def clock_data(signal_i, signal_q, sps, block_count):
    
	no_symbols = len(signal_i) // sps
	i_samples = np.zeros(no_symbols)
	q_samples = np.zeros(no_symbols)
	offset = sps // 2

	
	if(block_count >=10): ##wait for pll to lock before determining optimal sampling points 
			
			peaks = []

			for i in range(1, len(signal_i)-1):
				if signal_i[i] > signal_i[i-1] and signal_i[i] > signal_i[i+1]:
					peaks.append(i % sps)

			
			if peaks: 
				offset = int(np.median(peaks)) 


	
	for symbol_index in range(no_symbols):
		sample_pos = symbol_index * sps + offset 
		if sample_pos < len(signal_i):
			i_samples[symbol_index] = signal_i[sample_pos]
			q_samples[symbol_index] = signal_q[sample_pos]


	return i_samples, q_samples, offset

