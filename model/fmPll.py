#
# Comp Eng 3DY4 (Computer Systems Integration Project)
#
# Department of Electrical and Computer Engineering
# McMaster University
# Ontario, Canada
#

import numpy as np
import math
import matplotlib.pyplot as plt

def plotSpectrum(x, Fs, type = 'FFT'):

	n = len(x)             # length of the signal
	df = Fs/n              # frequency increment (width of freq bin)

	# compute Fourier transform, its magnitude and normalize it before plotting
	if type == 'FFT':
		Xfreq = np.fft.fft(x)
	#XMag = abs(Xfreq)/n
		
	if type == 'DFT':
		Xfreq = DFT(x)
    

	# Note: because x is real, we keep only the positive half of the spectrum
	# Note also: half of the energy is in the negative half (not plotted)
	XMag = abs(Xfreq)/n
	XMag = XMag[0:int(n/2)]

	# freq vector up to Nyquist freq (half of the sample rate)
	freq = np.arange(0, Fs/2, df)

	fig, ax = plt.subplots()
	ax.plot(freq, XMag)
	ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude',
		title='Frequency domain plot') 

def plotTime1(x, time):

	fig, ax = plt.subplots()

    # Plot the signal
	ax.plot(time, x)

    # Set labels and title
	ax.set(xlabel='Time (sec)', ylabel='Amplitude', title='Time domain plot')

	# Adjust the axis limits (e.g., make the y-axis range a bit larger for visibility)
	ax.set_xlim([0, 0.04])  # Set x-axis from the first to the last time sample
	ax.set_ylim([np.min(x) - 0.1, np.max(x) + 0.1])  # Set y-axis with a little margin

	fig.savefig("../data/plot.png")
	
	plt.show()
def plotTime(x1, x2, time):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # Create 2 subplots

    plt.plot(time, x1, label="Reference Signal (pllIn)", color='b', linestyle='-')
    plt.plot(time, x2, label="NCO Output (PLL)", color='g', linestyle='--')

    # Labels and title
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain Signals')
    
    # Adjust axis limits
    plt.xlim([np.min(time), np.max(time)])
    plt.ylim([min(np.min(x1), np.min(x2)) - 0.1, max(np.max(x1), np.max(x2)) + 0.1])

    # Grid and legend
    plt.grid()
    plt.legend()

    # Show the plot
    plt.show()





def fmPll(pllIn, freq, Fs, ncoScale = 2.0, phaseAdjust = 0.0, normBandwidth = 0.01):

	"""

	pllIn 	 		array of floats
					input signal to the PLL (assume known frequency)

	freq 			float
					reference frequency to which the PLL locks

	Fs  			float
					sampling rate for the input/output signals

	ncoScale		float
					frequency scale factor for the NCO output

	phaseAdjust		float
					phase adjust to be added to the NCO output only

	normBandwidth	float
					normalized bandwidth for the loop filter
					(relative to the sampling rate)

	state 			to be added

	"""

	# scale factors for proportional/integrator terms
	# these scale factors were derived assuming the following:
	# damping factor of 0.707 (1 over square root of 2)
	# there is no oscillator gain and no phase detector gain
	Cp = 2.666
	Ci = 3.555

	# gain for the proportional term
	Kp = (normBandwidth)*Cp

	# gain for the integrator term
	Ki = (normBandwidth*normBandwidth)*Ci

	# output array for the NCO
	ncoOut = np.empty(len(pllIn)+1)

	# initialize internal state
	integrator = 0.0
	phaseEst = 0.0
	feedbackI = 1.0
	feedbackQ = 0.0
	ncoOut[0] = 1.0
	trigOffset = 0

	# note: state saving will be needed for block processing
	for k in range(len(pllIn)):

		# phase detector
		errorI = pllIn[k] * (+feedbackI)  # complex conjugate of the
		errorQ = pllIn[k] * (-feedbackQ)  # feedback complex exponential

		# four-quadrant arctangent discriminator for phase error detection
		errorD = math.atan2(errorQ, errorI)

		# loop filter
		integrator = integrator + Ki*errorD

		# update phase estimate
		phaseEst = phaseEst + Kp*errorD + integrator

		# internal oscillator
		trigOffset += 1
		trigArg = 2*math.pi*(freq/Fs)*(trigOffset) + phaseEst
		feedbackI = math.cos(trigArg)
		feedbackQ = math.sin(trigArg)
		ncoOut[k+1] = math.cos(trigArg*ncoScale + phaseAdjust)

	# for stereo only the in-phase NCO component should be returned
	# for block processing you should also return the state

	return ncoOut


def fmPll_state(pllIn, freq, Fs, state, ncoScale = 2.0, phaseAdjust = 0.0, normBandwidth = 0.01):

	"""

	pllIn 	 		array of floats
					input signal to the PLL (assume known frequency)

	freq 			float
					reference frequency to which the PLL locks

	Fs  			float
					sampling rate for the input/output signals

	ncoScale		float
					frequency scale factor for the NCO output

	phaseAdjust		float
					phase adjust to be added to the NCO output only

	normBandwidth	float
	
					normalized bandwidth for the loop filter
					(relative to the sampling rate)

	state 			to be added

	"""

	# scale factors for proportional/integrator terms
	# these scale factors were derived assuming the following:
	# damping factor of 0.707 (1 over square root of 2)
	# there is no oscillator gain and no phase detector gain
	Cp = 2.666
	Ci = 3.555

	# gain for the proportional term
	Kp = (normBandwidth)*Cp

	# gain for the integrator term
	Ki = (normBandwidth*normBandwidth)*Ci

	# output array for the NCO
	ncoOut = np.empty(len(pllIn)+1)
	# initialize internal state
	integrator = state[0]
	phaseEst = state[1]
	feedbackI = state[2]
	feedbackQ = state[3]
	ncoOut[0] = state[4]
	trigOffset = state[5]
	feedbackI_array = np.empty(len(pllIn))  # Create an array to store feedbackI values

	# note: state saving will be needed for block processing
	for k in range(len(pllIn)):

		# phase detector
		errorI = pllIn[k] * (+feedbackI)  # complex conjugate of the
		errorQ = pllIn[k] * (-feedbackQ)  # feedback complex exponential

		# four-quadrant arctangent discriminator for phase error detection
		errorD = math.atan2(errorQ, errorI)

		# loop filter
		integrator = integrator + Ki*errorD

		# update phase estimate
		phaseEst = phaseEst + Kp*errorD + integrator

		# internal oscillator
		trigOffset += 1
		trigArg = 2*math.pi*(freq/Fs)*(trigOffset) + phaseEst
		feedbackI = math.cos(trigArg)
		feedbackQ = math.sin(trigArg)
		ncoOut[k+1] = math.cos(trigArg*ncoScale + phaseAdjust)
		feedbackI_array[k] = feedbackI  # Store feedbackI over time


	state[0] = integrator
	state[1] = phaseEst
	state[2] = feedbackI
	state[3] = feedbackQ
	state[4] = ncoOut[-1]
	state[5] = trigOffset + len(pllIn)


	# time = np.arange(len(pllIn)) / Fs
	# plotTime(feedbackI_array, ncoOut[:-1], time)

	# plotTime1(pllIn, time)



	# for stereo only the in-phase NCO component should be returned
	# for block processing you should also return the state

	return ncoOut, state


def fmPll_rds(pllIn, freq, Fs, state, ncoScale = 0.5, phaseAdjust = 0.0, normBandwidth = 0.005):

	"""

	pllIn 	 		array of floats
					input signal to the PLL (assume known frequency)

	freq 			float
					reference frequency to which the PLL locks

	Fs  			float
					sampling rate for the input/output signals

	ncoScale		float
					frequency scale factor for the NCO output

	phaseAdjust		float
					phase adjust to be added to the NCO output only

	normBandwidth	float
	
					normalized bandwidth for the loop filter
					(relative to the sampling rate)

	state 			to be added

	"""

	# scale factors for proportional/integrator terms
	# these scale factors were derived assuming the following:
	# damping factor of 0.707 (1 over square root of 2)
	# there is no oscillator gain and no phase detector gain
	Cp = 2.666
	Ci = 3.555

	# gain for the proportional term
	Kp = (normBandwidth)*Cp

	# gain for the integrator term
	Ki = (normBandwidth*normBandwidth)*Ci

	# output array for the NCO
	ncoOut = np.empty(len(pllIn)+1)
	ncoOut_q = np.empty(len(pllIn)+1)

	# initialize internal state
	integrator = state[0]
	phaseEst = state[1]
	feedbackI = state[2]
	feedbackQ = state[3]
	ncoOut[0] = state[4]
	ncoOut_q[0] = state[4]
	trigOffset = state[5]
	feedbackI_array = np.empty(len(pllIn))  # Create an array to store feedbackI values

	# note: state saving will be needed for block processing
	for k in range(len(pllIn)):

		# phase detector
		errorI = pllIn[k] * (+feedbackI)  # complex conjugate of the
		errorQ = pllIn[k] * (-feedbackQ)  # feedback complex exponential

		# four-quadrant arctangent discriminator for phase error detection
		errorD = math.atan2(errorQ, errorI)

		# loop filter
		integrator = integrator + Ki*errorD

		# update phase estimate
		phaseEst = phaseEst + Kp*errorD + integrator

		# internal oscillator
		trigOffset += 1
		trigArg = 2*math.pi*(freq/Fs)*(trigOffset) + phaseEst
		feedbackI = math.cos(trigArg)
		feedbackQ = math.sin(trigArg)
		ncoOut[k+1] = math.cos(trigArg*ncoScale + phaseAdjust)
		feedbackI_array[k] = feedbackI  # Store feedbackI over time
		ncoOut_q[k+1] = math.sin(trigArg*ncoScale + phaseAdjust)

	state[0] = integrator
	state[1] = phaseEst
	state[2] = feedbackI
	state[3] = feedbackQ
	state[4] = ncoOut[-1]
	state[5] = trigOffset + len(pllIn)


	#time = np.arange(len(pllIn)) / Fs
	#plotTime(feedbackI_array, ncoOut[:-1], time)

	#plotTime1(pllIn, time)



	# for stereo only the in-phase NCO component should be returned
	# for block processing you should also return the state

	return ncoOut,ncoOut_q, state

	# for RDS add also the quadrature NCO component to the output

if __name__ == "__main__":

	pass
