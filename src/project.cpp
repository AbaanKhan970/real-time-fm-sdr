/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include <cstdlib>
#include <vector>
#include "filter.h"
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>       
#include <functional>  


// Thread-safe queue
template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mtx;
    std::condition_variable cv;
    unsigned int max_size = 10; // our choice for queue size
    
public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this]() { return queue.size() < max_size; });
        queue.push(item);
        lock.unlock();
        cv.notify_one();
    }
    
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (queue.empty()) {
            return false;
        }
        item = queue.front();
        queue.pop();
        lock.unlock();
        cv.notify_one();
        return true;
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

void data(pllState &values)
{
	values.integrator = 0.0;
	values.phaseEst = 0.0;
	values.feedbackI = 1.0;
	values.feedbackQ = 0.0;
	values.ncoOut = 1.0;
	values.trigOffset = 0;
}

// queues for inter-thread communication
SafeQueue<std::vector<real>> rf_queue;      
//SafeQueue<std::vector<real>> rds_queue;     
SafeQueue<std::vector<short int>> audio_queue; 

// Thread control
std::atomic<bool> stop_threads(false);

void rf_frontend_thread(int rf_Fs, int rf_decim, const std::vector<real>& rf_coeff, std::vector<real>& state_i_lpf, std::vector<real>& state_q_lpf) 
{
    int block_size = int(rf_Fs * 40 / 1000);
    
    for (unsigned int block_id = 0; !stop_threads.load(); block_id++) 
	{
        std::vector<float> block_data(2 * block_size);

        readSTdinBlockData(2 * block_size, block_id, block_data);
        
        if ((std::cin.rdstate()) != 0) {
            std::cerr << "End of input stream reached" << std::endl;
            stop_threads.store(true);
            break;
        }

        // Process I/Q data / same code as RF Front-end before threading
        std::vector<real> temp_i_data(block_size);
        std::vector<real> temp_q_data(block_size);
        
        for (int k = 0, temp_data_index = 0; k < 2 * block_size; k += 2, temp_data_index++) {
            temp_i_data[temp_data_index] = block_data[k];
            temp_q_data[temp_data_index] = block_data[k + 1];
        }
        
        std::vector<real> i_filt_ds(temp_i_data.size() / rf_decim, 0.0);
        std::vector<real> q_filt_ds(temp_q_data.size() / rf_decim, 0.0);

        efficient_convolve_downsample(i_filt_ds, temp_i_data, rf_coeff, state_i_lpf, rf_decim);
        efficient_convolve_downsample(q_filt_ds, temp_q_data, rf_coeff, state_q_lpf, rf_decim);

        // Demodulate
        static real prev_I = 0.0;
        static real prev_Q = 0.0;
        std::vector<real> fmdemod;
        my_Demod(i_filt_ds, q_filt_ds, prev_I, prev_Q, fmdemod);

        // Push to required queues
        rf_queue.push(fmdemod);
        //rds_queue.push(fmdemod);
    }
}

// mono and stereo thread (audio thread)
void audio_thread(int mode, int rf_Fs, int rf_decim, int audio_Fs, int u, int d, const std::vector<real>& audio_coeff, const std::vector<real>& scr_bpf_coeff, const std::vector<real>& sce_bpf_coeff, int channel) 
{
    // States
    std::vector<real> state_audio_lpf(audio_coeff.size() - 1);
    std::vector<real> state_scr(scr_bpf_coeff.size() - 1);
    std::vector<real> state_sce(sce_bpf_coeff.size() - 1);
    std::vector<real> state_mixed(audio_coeff.size() - 1);
    std::vector<real> block_delay((scr_bpf_coeff.size() - 1) / 2, 0.0);

    pllState state_pll;
    data(state_pll);
    
    std::vector<real> fmdemod;
    
    while (!stop_threads.load()) {
        if (!rf_queue.pop(fmdemod)) {
            if (stop_threads.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // wait for loading
            continue;
        }

        std::vector<real> audio_block((u / d) * fmdemod.size());
        std::vector<real> mixed_filt((fmdemod.size() * u) / d, 0.0);
        std::vector<real> audio_full;

        if (mode == 0 || mode == 1) 
		{
            if (channel == 1) 
			{                
                std::vector<real> scr_filt, nco_out;
                efficient_convolve_downsample(scr_filt, fmdemod, scr_bpf_coeff, state_scr, 1);
                fmPll(scr_filt, 19e3, rf_Fs / rf_decim, state_pll, 2.0, 0.0, 0.01, nco_out);

				std::vector<real> sce_filt;
                efficient_convolve_downsample(sce_filt, fmdemod, sce_bpf_coeff, state_sce, 1);

                std::vector<real> mixed(fmdemod.size());
                for (int i = 0; i < (int)fmdemod.size(); i++) 
				{
                    mixed[i] = sce_filt[i] * nco_out[i];
                }

                efficient_convolve_downsample(mixed_filt, mixed, audio_coeff, state_mixed, d);
                
				for (int i = 0; i < (int)mixed_filt.size(); i++) 
				{
                    mixed_filt[i] *= 2;
                }

                std::vector<real> delayed_block;
                delay_block(fmdemod, delayed_block, block_delay);

                efficient_convolve_downsample(audio_block, delayed_block, audio_coeff, state_audio_lpf, d);

                audio_full.resize(mixed_filt.size() * 2);
                for (int i = 0; i < (int)audio_block.size(); i++) 
				{
                    audio_full[2 * i] = (audio_block[i] + mixed_filt[i]);
                    audio_full[(2 * i) + 1] = (audio_block[i] - mixed_filt[i]);
                }
            } 
			else 
			{
                efficient_convolve_downsample(audio_full, fmdemod, audio_coeff, state_audio_lpf, d);
            }
        } 
		else if (mode == 2 || mode == 3) 
		{
            if (channel == 1) 
			{
                std::vector<real> scr_filt, nco_out;
                efficient_convolve_downsample(scr_filt, fmdemod, scr_bpf_coeff, state_scr, 1);
                fmPll(scr_filt, 19e3, rf_Fs / rf_decim, state_pll, 2.0, 0.0, 0.01, nco_out);

				std::vector<real> sce_filt;
                efficient_convolve_downsample(sce_filt, fmdemod, sce_bpf_coeff, state_sce, 1);

                std::vector<real> mixed(fmdemod.size());

                for (int i = 0; i < (int)fmdemod.size(); i++) 
				{
                    mixed[i] = sce_filt[i] * nco_out[i];
                }

                resampler(mixed_filt, mixed, audio_coeff, state_mixed, d, u);

                for (int i = 0; i < (int)mixed_filt.size(); i++) 
				{
                    mixed_filt[i] *= 2 * u;
                }

                std::vector<real> delayed_block;
                delay_block(fmdemod, delayed_block, block_delay);

                resampler(audio_block, delayed_block, audio_coeff, state_audio_lpf, d, u);

                for (int i = 0; i < audio_block.size(); i++) 
				{
                    audio_block[i] *= u;
                }

                audio_full.resize(mixed_filt.size() * 2);
                for (int i = 0; i < (int)audio_block.size(); i++) 
				{
                    audio_full[2 * i] = (audio_block[i] + mixed_filt[i]);
                    audio_full[(2 * i) + 1] = (audio_block[i] - mixed_filt[i]);
                }
            } 
			else 
			{
                // Mono processing with resampling
                resampler(audio_full, fmdemod, audio_coeff, state_audio_lpf, d, u);
                for (int i = 0; i < audio_full.size(); i++) 
				{
                    audio_full[i] *= u;
                }
            }
        }


        std::vector<short int> audio_data(audio_full.size());
        for (unsigned int k = 0; k < audio_full.size(); k++) 
		{
            audio_data[k] = std::isnan(audio_full[k]) ? 0 : static_cast<short int>(audio_full[k] * 8192);
        }
        audio_queue.push(audio_data);
    }
}

/*
// RDS thread function (to be implemented)
void rds_thread(int mode, int rf_Fs, int rf_decim) {
    // TODO: Implement RDS processing
    while (!stop_threads.load()) {
        std::vector<real> fmdemod;
        if (!rds_queue.pop(fmdemod)) {
            if (stop_threads.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // wait for loading
            continue;
        }
        
        // RDS processing would go here
    }
}
*/
int main(int argc, char *argv[]) {
    //NOTE
    //===============================================================================
    //RUN THE UNIX PIPE WITH NO ARGUEMENTS FOR MODE AND CHANNEL OR ARGUEMENTS FOR BOTH
    //==================================================================================

    int mode = modeSelect(argc, argv);
    int channel = channelSelect(argc, argv);
    //int channel = 0;

    int rf_Fs = 2400e3; 
	int rf_Fc = 100e3; 
	int rf_taps = 101; 
	int rf_decim = 10;
    
	int audio_Fs = 48e3; 
	int audio_Fc = 16e3; 
	int audio_taps = 101; 
	int u = 1; 
	int d = 5;
    
    if (mode == 0) {
        rf_Fs = 2400e3; 
		rf_decim = 10;
        audio_Fs = 48e3; 
		u = 1; 
		d = 5;
    } else if (mode == 1) {
        rf_Fs = 2880e3; 
		rf_decim = 8;
        audio_Fs = 36e3; 
		u = 1; 
		d = 10;
    } else if (mode == 2) {
        rf_Fs = 2400e3; 
		rf_decim = 10;
        audio_Fs = 44.1e3; 
		u = 147; 
		d = 800;
        audio_taps = 101 * u;
    } else if (mode == 3) {
        rf_Fs = 960e3; 
		rf_decim = 5;
        audio_Fs = 44.1e3; 
		u = 147; 
		d = 640;
        audio_taps = 101 * u;
    } 
	else 
	{
        std::cerr << "Invalid mode selected" << std::endl;
        return 1;
    }

    // filter coefficients
    std::vector<real> rf_coeff;
    impulseResponseLPF(rf_Fs, rf_Fc, rf_taps, rf_coeff);

	std::vector<real> audio_coeff;
    impulseResponseLPF(u * (rf_Fs / rf_decim), audio_Fc, audio_taps, audio_coeff);

    // Stereo
    int Flow_scr = 18.5e3;
	int Fhigh_scr = 19.5e3;
    int Flow_sce = 22e3; 
	int Fhigh_sce = 54e3;

    std::vector<real> scr_bpf_coeff;
    impulseResponseBPF(Flow_scr, Fhigh_scr, (rf_Fs / rf_decim), audio_taps / u, scr_bpf_coeff);

	std::vector<real> sce_bpf_coeff;
    impulseResponseBPF(Flow_sce, Fhigh_sce, (rf_Fs / rf_decim), audio_taps / u, sce_bpf_coeff);

    // States
    std::vector<real> state_i_lpf(rf_taps - 1);
    std::vector<real> state_q_lpf(rf_taps - 1);

    // launch threads
    std::thread rf_thread(rf_frontend_thread, rf_Fs, rf_decim, std::ref(rf_coeff), std::ref(state_i_lpf), std::ref(state_q_lpf));
    std::thread audio_processor(audio_thread, mode, rf_Fs, rf_decim, audio_Fs, u, d, std::ref(audio_coeff), std::ref(scr_bpf_coeff), std::ref(sce_bpf_coeff), channel);
    //std::thread rds_processor(rds_thread, mode, rf_Fs, rf_decim);

    // Main thread handles audio output
    while (!stop_threads.load()) {
        std::vector<short int> audio_data;
        if (audio_queue.pop(audio_data)) {
            if (!audio_data.empty()) {
                fwrite(&audio_data[0], sizeof(short int), audio_data.size(), stdout);
            }
        } else {
            if (stop_threads.load()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));  // wait for loading
        }
    }

    // Clean up
    rf_thread.join();
    audio_processor.join();
    //rds_processor.join();

    return 0;
}