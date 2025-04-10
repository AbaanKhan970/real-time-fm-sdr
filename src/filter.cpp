/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include "math.h"




//IMPULSE RESPONSE
// Function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(real Fs, real Fc, unsigned short int num_taps, std::vector<real> &h)
{
	// Bring your own functionality
    // Allocate memory for the impulse response
	h.clear();
	h.resize(num_taps, 0.0);

	// The rest of the code in this function is to be completed by you
	// based on your understanding and the Python code from the first lab
	real Norm_c = Fc/(Fs/2.0);
	real Middle_i = (num_taps - 1)/2.0;

	//h = np.zeros(N_taps);
	real numerator, denominator;

	for (int i=0; i<num_taps; i++){
		if (i == (num_taps - 1)/2) {
			h[i] = Norm_c;
		}
		else{
			//h[i] = Norm_c * sinc(Norm_c*(i-Middle_i));
			numerator = std::sin(PI * Norm_c * (i - Middle_i));
			denominator = PI * Norm_c * (i - Middle_i);
			h[i] = Norm_c * (numerator / denominator);
		}
		h[i] *= ((1/2.0) - ((1/2.0) * std::cos((2 * PI * i)/(num_taps-1))));
        
    }
}

//CONVOLUTION 
void convolveFIR(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state) {    // Allocate memory for the output (filtered) data
    //to be used as reference for unit testing
	y.clear();
	y.resize(x.size(), 0);

    for(int i = 0; i < y.size(); i++){
		y[i] = 0;
		
        for(int k = 0; k < h.size(); k++){
             
            if ((i  - k) >= 0){
                y[i] += h[k] * x[(i - k)];
				
            }
            else {
                
                y[i] += h[k] * state[state.size() + (i-k)];
            }
		}
    }

	std::copy(x.begin() + (x.size() - (h.size()- 1)), x.end(), state.begin());
}


void slow_convolve_downsample(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim, const int u) {    // Allocate memory for the output (filtered) data
    
	// upsample
	//   pad with zeros
	std::vector<real> upsampled(u * x.size(), 0);
	for (int j = 0; j < x.size(); j++) {
			upsampled[j * u] = x[j];
	}

	// convolve
	//   regular convolution with state saving

	convolveFIR(y,x,h,state);
	int count = 0;
	for (int p = 0; p < state.size(); p+=u)
	{

		state[count] = state[p];
		count++;
	}
	state.resize(state.size()/u);

	// downsample
	//   discard values
	int count2 = 0 ;
	for (int q = 0; q < y.size(); q+=decim)
	{

		y[count2] = y[q];
		count2++;
	}
	y.resize(y.size()/decim);


	// filter gain
	// for (int i = 0; i < y.size(); i++) {
				// y[i] *= u;
	// }
}


void resampler(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim, const int up) {    // Allocate memory for the output (filtered) data
    y.clear();
	y.resize(x.size()*up/decim, 0.0);
	for (int n = 0; n < y.size(); n++){
		y[n] = 0;	
		int phase = (n*decim) % up;
		for(int k = phase; k < h.size(); k += up){
			int idx = (n * decim - k) / up;
			if(idx >= 0){
				y[n] += h[k] * x[idx];
			}
			else{
				y[n] += h[k] * state[state.size() + idx];
			}
		}
	}
	std::copy(x.begin() + (x.size() - (h.size() - 1)), x.end(), state.begin());
}

//CONVOLUTION AND DEMODULATION
void efficient_convolve_downsample(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim) {    // Allocate memory for the output (filtered) data
    y.clear();
	y.resize((x.size()/decim), 0.0);

    for(int i = 0; i < y.size(); i++){
		y[i] = 0;
        for(int k = 0; k < h.size(); k++){
             
            if ((i*decim  - k) >= 0){
                y[i] += h[k] * x[(i*decim - k)];
				
            }
            else {
                
                y[i] += h[k] * state[state.size() + ((i*decim-k))];
            }
		}
    }

	std::copy(x.begin() + (x.size() - (h.size() - 1)), x.end(), state.begin());
}

//DEMODULATION
void my_Demod(const std::vector<real> &I, const std::vector<real> &Q, real &prev_I, real &prev_Q, std::vector<real> &fm_demod) {


	fm_demod.resize(I.size());


	for (int k=0; k<int(I.size()); k++)
    {
        if((I[k]*I[k]) + (Q[k]*Q[k])  == 0)
        {
            fm_demod[k] =  0;
        }
			
		else
        {
            real numerator =  (I[k]*(Q[k]-prev_Q)) - (Q[k]*(I[k]-prev_I));
			real denominator =  ((I[k]*I[k]) + (Q[k]*Q[k]));
			fm_demod[k] = numerator/denominator;
        }

		prev_I = I[k];
		prev_Q = Q[k];
    }
		
}

//FUNCTION FOR SELECTION OF MODE
int modeSelect(int argc, char *argv[]){
	int mode = 0;

	if (argc < 2){
		std::cerr << "Operating in default mode 0" << std::endl;
	}

	else if (argc >= 2 ){
		mode = atoi(argv[1]);
		if (mode > 3){
			std::cerr << "Wrong mode...exiting" << std::endl;
			exit(1);
		}

		else if (mode == 1)
		{
			std::cerr << "Operating in mode 1" << std::endl;
		}

		else if (mode == 2)
		{
			std::cerr << "Operating in mode 2" << std::endl;
		}

		else if (mode == 3)
		{
			std::cerr << "Operating in mode 3" << std::endl;
		}
	}

	else{
		std::cerr << "select a correct mode between 0 and 3" << std::endl;
	}

	return mode;
}

int channelSelect(int argc, char *argv[]){
	std::string channel;
	int channel_int = 0;
	channel = (argv[2]);
	if (channel == "m") {
		channel_int	= 0;	
		std::cerr << "Channel: Mono selected" << std::endl;
	} else if (channel == "s") {
		channel_int	= 1;
		std::cerr << "Channel: Stereo selected" << std::endl;
	} else if (channel == "r") {
		channel_int	= 2;
		std::cerr << "Channel: RDS selected" << std::endl;
	}
	return channel_int;
}


/////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void convolveFIR_inefficient(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    y.clear();
    y.resize(int(x.size() + h.size()-1), 0.0);
    for (auto n = 0; n < (int)y.size(); n++) {
        for (auto k = 0; k < (int)x.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)h.size())
                y[n] += x[k] * h[n - k];
        }
    }
}

void convolveFIR_reference(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    std::vector<real> filterResult(int(x.size() + h.size()-1), 0.0);
	filterResult.clear();
	y.clear();

    for (auto n = 0; n < (int)filterResult.size(); n++) {
        for (auto k = 0; k < (int)h.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)x.size())
                filterResult[n] += h[k] * x[n - k];
        }
    }

	// //downsampling
    // for(int k = 0; k < filterResult.size(); k+=decim)
	// {
	// 		y.push_back(filterResult[k]);			
	// }
	
}


void fmPll(std::vector<real> &pllIn, real freq, real Fs, pllState &state, real ncoScale, real phaseAdjust, real normBandwidth, std::vector<real> &ncoOut) {
	const real Cp = 2.666;
	const real Ci = 3.555;

	real Kp = (normBandwidth)*Cp;
	real Ki = (normBandwidth*normBandwidth)*Ci;

	ncoOut.resize(pllIn.size()+1);

	real integrator = state.integrator;
	real phaseEst = state.phaseEst;
	real feedbackI = state.feedbackI;
	real feedbackQ = state.feedbackQ;
	ncoOut[0] = state.ncoOut;
	int trigOffset = state.trigOffset;


	for (int k = 0; k < (int)pllIn.size(); k++) {

		real error_I = pllIn[k] * (+feedbackI);
        real error_Q = pllIn[k] * (-feedbackQ);

        real error_D = std::atan2(error_Q, error_I);

        integrator += Ki * error_D;
        phaseEst += (Kp * error_D) + integrator;

        trigOffset += 1;
        real trig_arg = 2.0 * PI * (freq / Fs) * (trigOffset) + phaseEst;
        feedbackI = std::cos(trig_arg);
        feedbackQ = std::sin(trig_arg);
        ncoOut[k + 1] = std::cos((trig_arg * ncoScale) + phaseAdjust);
    }


	state.integrator = integrator;
	state.phaseEst = phaseEst;
	state.feedbackI = feedbackI;
	state.feedbackQ = feedbackQ;
	state.ncoOut = ncoOut.back();	
	state.trigOffset = trigOffset + pllIn.size();


}


void delay_block(const std::vector<real> &input, std::vector<real> &output, std::vector<real> &state) {
    output.clear();
    //state.clear();
    
    for (int i = 0; i < (int)state.size(); i++) { 
        output.push_back(state[i]);
    }

    for (unsigned int i = 0; i < (int)input.size() - state.size(); i++) { 
        output.push_back(input[i]);
    }

    state.assign(input.end() - state.size(), input.end());
}

void impulseResponseBPF(real Flow, real Fhigh, real Fs, unsigned short int N_taps, std::vector<real> &h)
{
	// Bring your own functionality
    // Allocate memory for the impulse response
	h.clear();
	h.resize(N_taps, 0.0);

	// The rest of the code in this function is to be completed by you
	// based on your understanding and the Python code from the first lab
	int center = (N_taps-1) / 2;
	real Fmid = (Fhigh + Flow) / 2.0;

	real scale_factor = 0.0;

	for (int k = 0; k < N_taps; k++){
		int n = k - center;
		if (n == 0){
			h[k] = 2.0 * ((Fhigh - Flow) / Fs);
		}
		else{
			h[k] = (std::sin((2.0 * PI * n * Fhigh) / Fs) - std::sin((2.0 * PI * n * Flow) / Fs)) / (PI * n);
		}
		h[k] = h[k] * real(0.5 - (0.5 * std::cos((2.0 * PI * k) / (N_taps - 1))));
		scale_factor += h[k] * real(std::cos((2.0 * PI * n * Fmid) / Fs));
    }
	for (int k = 0; k < N_taps; k++){
		h[k] = real(h[k] / scale_factor);
	}
}


