/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#ifndef DY4_FILTER_H
#define DY4_FILTER_H

// Add headers as needed
#include <iostream>
#include <vector>

struct pllState{
    float integrator, phaseEst, feedbackI, feedbackQ, ncoOut;
    int trigOffset;
};

//FOR PROJECT
// Declaration of function prototypes
int modeSelect(int argc, char *argv[]);
int channelSelect(int argc, char *argv[]);



void impulseResponseLPF(real Fs, real Fc, unsigned short int num_taps, std::vector<real> &h);
void convolveFIR(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state);
void my_Demod(const std::vector<real> &I, const std::vector<real> &Q, real &prev_I, real &prev_Q, std::vector<real> &fm_demod);
void efficient_convolve_downsample(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim) ;
void slow_convolve_downsample(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim, const int u);
void resampler(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h, std::vector<real> &state,  const int decim, const int up)  ;   

//////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void convolveFIR_inefficient(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h);
void convolveFIR_reference(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h);


//BPF
void impulseResponseBPF(real Flow, real Fhigh, real Fs, unsigned short int N_taps, std::vector<real> &h);
//void impulseResponseBPF(real Fb, real Fe, real Fs, int n_taps, std::vector<real> &h);

//delay
void delay_block(const std::vector<real> &input, std::vector<real> &output, std::vector<real> &state);

//pll  
void fmPll(std::vector<real> &pllIn, real freq, real Fs, pllState &state, real ncoScale, real phaseAdjust, real normBandwidth, std::vector<real> &ncoOut);

#endif // DY4_FILTE

