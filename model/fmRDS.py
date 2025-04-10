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
from fmSupportLib import fmDemodArctan, fmPlotPSD, my_Demod, my_own_coeff, my_own_coeffbpf, my_own_convolution, my_own_convolution_ns, block_delay, resampler, plot_rrc_with_offsets, clock_data
from fmPll import fmPll, fmPll_state, plotTime, fmPll_rds
from fmRRC import impulseResponseRootRaisedCosine
# for take-home add your functions

# functions for application layer

def extract_pi_code_bitwise(block_bits):
    """Extract PI code from Block 1 (bits 0-15 of the 26-bit block)."""
    pi_bits = block_bits[0:16]  # First 16 bits are PI code
    pi_code = 0
    for bit in pi_bits:
        pi_code = (pi_code << 1) | int(bit)
    return pi_code

def extract_pty_code_bitwise(block_bits):
    """Extract PTY code from Block 2 (bits 11-15 of the 26-bit block)."""
    pty_bits = block_bits[11:16]  # PTY is bits 11-15 (see RBDS standard)
    pty_code = 0
    for bit in pty_bits:
        pty_code = (pty_code << 1) | int(bit)
    return pty_code

def extract_group_type_bitwise(block_bits):
    """Extract group type (bits 0-3) and version (bit 4) from Block 2."""
    group_type_bits = block_bits[0:4]  # Lower nibble (bits 0-3)
    version_bit = block_bits[4]        # Bit 4 (0=A, 1=B)
    
    group_type = 0
    for bit in group_type_bits:
        group_type = (group_type << 1) | int(bit)
    
    version = 'B' if version_bit else 'A'
    return f"{group_type}{version}"  # e.g., "0A" or "2B"

def extract_ps_name_bitwise(blocks_bits):
    """Extract Program Service name (8 ASCII chars from Blocks 1-4)."""
    ps_chars = []
    for block_idx in range(4):  # Blocks 1-4 contribute 2 chars each
        # PS chars are in bits 16-31 of each block (see RBDS standard)
        char1_bits = blocks_bits[block_idx*26 + 16 : block_idx*26 + 24]
        char2_bits = blocks_bits[block_idx*26 + 24 : block_idx*26 + 32]
        
        char1 = chr(int(''.join(map(str, char1_bits)), 2))
        char2 = chr(int(''.join(map(str, char2_bits)), 2))
        ps_chars.extend([char1, char2])
    
    return ''.join(ps_chars).strip()



def decode_group_bitwise(group_bits):
    """Decode a full RDS group (104 bits) directly."""
    if len(group_bits) != 104 or None in group_bits:
        return None

    # Split into 4 blocks (26 bits each)
    blocks = [
        group_bits[0:26],   # Block 1
        group_bits[26:52],  # Block 2
        group_bits[52:78],  # Block 3
        group_bits[78:104]  # Block 4
    ]

    result = {
        'pi_code': extract_pi_code_bitwise(blocks[0]),
        'pty_code': extract_pty_code_bitwise(blocks[1]),
        'group_type': extract_group_type_bitwise(blocks[1]),
    }

    # Handle group-specific data
    if result['group_type'] == '0A':
        result['ps_name'] = extract_ps_name_bitwise(blocks)
    # elif result['group_type'] == '2A':
    #     result['radio_text'] = extract_radio_text_bitwise(blocks)
    
    return result

# flag for mode 0 (0) vs mode 1 (1) 
channel = 1 #mono: 0, stereo: 1, rds: 2
mode = 0

#RDS BPF Coefficients
Flow_rds_ce = 54e3
Fhigh_rds_ce = 60e3

Flow_rds_cr = 113.5e3
Fhigh_rds_cr = 114.5e3

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
        
    sps = 20

    rds_u = 19
    rds_d = 96

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
    in_fname = "../data/iq_samples.raw"
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
    rf_coeff = signal.firwin(rf_taps, rf_Fc/(rf_Fs/2), window=('hann'))

    # coefficients for the filter to extract mono audio


    Fs_calculated = rf_Fs / rf_decim
    

    

    rds_ce_coeff = my_own_coeffbpf(Flow_rds_ce, Fhigh_rds_ce, Fs_calculated, rf_taps)
    rds_cr_coeff = my_own_coeffbpf(Flow_rds_cr, Fhigh_rds_cr, Fs_calculated, rf_taps)


    #rds_ce_coeff = signal.firwin(rf_taps, [Flow_rds_ce/(audio_Fs/2),Fhigh_rds_ce/(audio_Fs/2)], window=('hann'), pass_zero="bandpass")
    #rds_cr_coeff = signal.firwin(rf_taps, [113.5e3/(audio_Fs/2),114.5e3/(audio_Fs/2)], window=('hann'), pass_zero="bandpass")



    #rds_demod_coeff = my_own_coeff(Fs_calculated, 3e3, rf_taps)
    rds_demod_coeff = signal.firwin(rf_taps,3000/(240000/2), window=('hann'))

    rds_resample_coeff = my_own_coeff(Fs_calculated * rds_u, 3e3, audio_taps * rds_u)
    #rds_resample_coeff = signal.firwin(rf_taps, (47500/2)/((240000*rds_u)/2), window=('hann'))


    # rds_resample_coeff = signal.firwin(rf_taps * rds_u, 240, window=('hann'))

    #RRC
    rrc_coeff = impulseResponseRootRaisedCosine(47500, rf_taps)

    # # set up the subfigures for plotting
    # subfig_height = np.array([2, 2, 2]) # relative heights of the subfigures
    # plt.rc('figure', figsize=(7.5, 7.5))	# the size of the entire figure
    # fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': subfig_height})
    # fig.subplots_adjust(hspace = .6)

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

    

    symbols_i_full = np.array([])
    symbols_q_full = np.array([])

    state_rds_cr = np.zeros(rf_taps - 1)
    state_rds_ce = np.zeros(rf_taps - 1)

    state_rds_demod_i = np.zeros(rf_taps - 1)
    state_rds_demod_q = np.zeros(rf_taps - 1)

    state_resample_i = np.zeros((rf_taps * rds_u) - 1)
    state_resample_q = np.zeros((rf_taps * rds_u) - 1)

    state_i = np.zeros(rf_taps - 1)
    state_q = np.zeros(rf_taps - 1)

    state_rrc_i = np.zeros(rf_taps - 1)
    state_rrc_q = np.zeros(rf_taps - 1)

    state_pll = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0])


    demod_i_full = np.array([])
    demod_q_full = np.array([])

    ncoOut_full = np.array([])
    i_samples = np.array([])
    q_samples = np.array([])

    #plotting
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(nrows=4)
    fig.subplots_adjust(hspace = 1.0)

    last_symbol = 0
    error1 = 0
    error2 = 0
    manchester_state = 0

    diff_stream = np.array([])

    position=0

    a_found = False
    b_found = False
    c_found = False
    d_found = False
    current_expected = None 
    current_group = [None] * 104
    # if the number of samples in the last block is less than the block size
    # it is fine to ignore the last few samples from the raw IQ file

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

        fm_demod, state_phase = fmDemodArctan(i_ds, q_ds, state_phase)

               
        

        #print('len fm_demod: ', len(fm_demod))
        #fm_demod_full = np.concatenate((fm_demod_full, fm_demod))

        if (mode == 0):

            #RDS Channel Extraction  
            rds_ce_filt, state_rds_ce = my_own_convolution(fm_demod, rds_ce_coeff, state_rds_ce, 1)
            # print('RDS CE DONE ')
            #print('len rds_ce_filt: ', len(rds_ce_filt))



            delay_block = np.array([0.0] * ((rf_taps - 1) // 2))
            delayed_audio_block, delay_block = block_delay(rds_ce_filt, delay_block)

            
            #RDS Carrier Recovery 
            #Squaring non-linearity
            rds_ce_filt_sqr = np.zeros(len(rds_ce_filt))
            for i in range(len(rds_ce_filt)):
                rds_ce_filt_sqr[i] = rds_ce_filt[i] * rds_ce_filt[i]


            
            #print('squaring DONE ')
            #print('len squared: ', len(rds_ce_filt_sqr))

            #BPF            
            rds_cr_filt, state_rds_cr = my_own_convolution(rds_ce_filt_sqr, rds_cr_coeff, state_rds_cr, 1)
                     
            
            ncoOut, ncoOut_q, state_pll = fmPll_rds(rds_cr_filt, 114e3, Fs_calculated, state_pll, 0.5, 0.0, 0.003)
            
            ncoOut, ncoOut_q, state_pll = fmPll_rds(rds_cr_filt, 114e3, Fs_calculated, state_pll, 0.5, 0.0, 0.003)



            #print('pll AND Nco DONE ')
            #print('len ncoOut: ', len(ncoOut))
            #print('len ncoOut_q: ', len(ncoOut_q))

            # ax1.psd(ncoOut, NFFT=512, Fs=(rf_Fs/rf_decim)/1e3)
            # ax1.set_ylabel('PSD (db/Hz)')
            # ax1.set_title('Recovery pll')

            
            #MIXER
        
            mixed_i = np.array([0.0] * len(delayed_audio_block))
            mixed_q = np.array([0.0] * len(delayed_audio_block))
            
            #mixed_q[i] = delayed_audio_block[i] * ncoOut_q[i] * 2

            for i in range(len(delayed_audio_block)):
                    mixed_i[i] = delayed_audio_block[i] * ncoOut[i] * 2
                    mixed_q[i] = delayed_audio_block[i] * ncoOut_q[i] * 2

            #print('mixer DONE ')
            #print('len mixed: ', len(mixed_q))
            #RDS DEMOD  
            #LPF
            
            rds_demod_i, state_rds_demod_i = my_own_convolution(mixed_i, rds_demod_coeff, state_rds_demod_i, 1)
            rds_demod_q, state_rds_demod_q = my_own_convolution(mixed_q, rds_demod_coeff, state_rds_demod_q, 1)


            #print('rds lpf DONE ')
            #print('len rds_demod: ', len(rds_demod_i))

            ax2.psd(rds_demod_i, NFFT=512, Fs=(rf_Fs/rf_decim)/1e3)
            ax2.set_ylabel('PSD (db/Hz)')
            ax2.set_title('Mixed rds Inphase')
            #plt.show()

            #resampler            
                    
            
            # Upsample
            upsampled_i = np.zeros(len(rds_demod_i) * rds_u)
            upsampled_q = np.zeros(len(rds_demod_q) * rds_u)
            for i in range(len(rds_demod_i)):
                upsampled_i[i*rds_u] = rds_demod_i[i]
                upsampled_q[i*rds_u] = rds_demod_q[i]

            # Filter with optimized convolution (no decimation here)
            filtered_i, state_resample_i = my_own_convolution(upsampled_i, rds_resample_coeff, state_resample_i, rds_d)
            filtered_q, state_resample_q = my_own_convolution(upsampled_q, rds_resample_coeff, state_resample_q, rds_d)

            # Downsample
            rds_resampled_i = filtered_i[::rds_d] * rds_u
            rds_resampled_q = filtered_q[::rds_d] * rds_u



            #print('resampler DONE ')
            #print('len rds_resampled: ', len(rds_resampled_i))
                    

            rrc_i, state_rrc_i = my_own_convolution(rds_resampled_i, rrc_coeff, state_rrc_i, 1)
            rrc_q, state_rrc_q = my_own_convolution(rds_resampled_q, rrc_coeff, state_rrc_q, 1)

        
            

            i_samples, q_samples, offset = clock_data(rrc_i, rrc_q, sps, block_count)
            print("Offset: ", offset)

            #if block_count >= 10:
                #plot_rrc_with_offsets(rrc_i, sps, offset)

            # this_i_samples = np.array([])

            # while offset < len(rrc_i):
            #     i_samples = np.concatenate((i_samples, [rrc_i[offset]]))
            #     this_i_samples = np.concatenate((this_i_samples, [rrc_i[offset]]))
            #     q_samples = np.concatenate((q_samples, [rrc_q[offset]]))
            #     offset += sps


            if(block_count == 246):
                plt.figure(figsize=(6, 6))
                plt.scatter(i_samples, q_samples, s=10)
                plt.title("IQ Constellation")
                plt.xlabel("In-phase (I)")
                plt.ylabel("Quadrature (Q)")
                plt.grid(True)
                plt.xlim(-1.5, 1.5)
                plt.ylim(-1.5, 1.5)
                plt.gca().set_aspect('equal', 'box')
                plt.tight_layout()
                plt.show()

            

            if block_count < 10:
                for i in range (0,len(i_samples), 2):
                    current = (i_samples[i] > 0)
                    previous = (i_samples[i-1]>0)
                    current_1 = (i_samples[i-1]>0)
                    if(i != 1):
                        previous_1 = (i_samples[i-2] > 0)
                    
                    else:
                        previous_1 = (symbols_state>0)

                    if (current == previous):
                        error1 += 1
                    if (current_1 == previous_1):
                        error2 += 1

                symbols_state = (i_samples[-1]>0)

            else:
                if error1 > error2:
                    decode_type = 0
                else:
                    decode_type = 1
                print("decode type: ", decode_type)

                manchester = []
                differential = []

                for i in range (decode_type,len(i_samples), 2):

                    if (i != 0):
                        prev = i_samples[i-1]
                    else:
                        prev = symbols_state

                    current = i_samples[i]

                    # High to low
                    if (prev > 0 and current < 0):
                        manchester.append(1) 
                        if (manchester_state == 0):	
                            differential.append(1) 
                        else:
                            differential.append(0) 
                    # Low to High
                    elif (prev < 0 and current > 0):
                        manchester.append(0) 
                        if (manchester_state == 1):	
                            differential.append(1) 
                        else:
                            differential.append(0) 
                    #print("man: ", manchester)
                    if manchester:
                        manchester_state = manchester[-1]
                symbols_state = (i_samples[-1]>0)


                #print("man: ", manchester)
                #print("diff: ", diff_stream)  
                if differential:
                    diff_stream = np.concatenate((diff_stream, np.array(differential, dtype=np.int8)))



           
             
            parity = np.matrix([
                                        [1,0,0,0,0,0,0,0,0,0],
                                        [0,1,0,0,0,0,0,0,0,0],
                                        [0,0,1,0,0,0,0,0,0,0],
                                        [0,0,0,1,0,0,0,0,0,0],
                                        [0,0,0,0,1,0,0,0,0,0],
                                        [0,0,0,0,0,1,0,0,0,0],
                                        [0,0,0,0,0,0,1,0,0,0],
                                        [0,0,0,0,0,0,0,1,0,0],
                                        [0,0,0,0,0,0,0,0,1,0],
                                        [0,0,0,0,0,0,0,0,0,1],
                                        [1,0,1,1,0,1,1,1,0,0],
                                        [0,1,0,1,1,0,1,1,1,0],
                                        [0,0,1,0,1,1,0,1,1,1],
                                        [1,0,1,0,0,0,0,1,1,1],
                                        [1,1,1,0,0,1,1,1,1,1],
                                        [1,1,0,0,0,1,0,0,1,1],
                                        [1,1,0,1,0,1,0,1,0,1],
                                        [1,1,0,1,1,1,0,1,1,0],
                                        [0,1,1,0,1,1,1,0,1,1],
                                        [1,0,0,0,0,0,0,0,0,1],
                                        [1,1,1,1,0,1,1,1,0,0],
                                        [0,1,1,1,1,0,1,1,1,0],
                                        [0,0,1,1,1,1,0,1,1,1],
                                        [1,0,1,0,1,0,0,1,1,1],
                                        [1,1,1,0,0,0,1,1,1,1],
                                        [1,1,0,0,0,1,1,0,1,1],
                                        ])
            sync = False
            
            syndrome =""
            if position+26 > len(diff_stream):
                sync = True
            
            print("pos: ", position)
            #print("diff: ", diff_stream)  
            print("len diff: ", len(diff_stream))  

            while sync == False:
                check_block = diff_stream[position:position+26]
                #print("check_block: ", check_block)  
                #print("len check_block: ", len(check_block))  
                result = np.zeros(10)
                ## create 1x10 matrix via bitwise matrix multiplication 
                for i in range(len(result)):
                    for j in range(26):
                        # AND MULTIPLICATION
                        mat_mult = check_block[j] and parity[j, i]
                        # XOR ADDITION
                        result[i] = (result[i] and not mat_mult) or (not result[i] and mat_mult)
                
                result = result.astype(int)
                result = (result).tolist()

                
                if current_expected is None:
                    
                    if result == [1,1,1,1,0,1,1,0,0,0]:
                        syndrome = "A"
                        print(f"Syndrome is A at position: {position+26}")
                        current_expected = "B"  # Next we expect B
                    elif result == [1,1,1,1,0,1,0,1,0,0]:
                        syndrome = "B"
                        print(f"Syndrome is B at position: {position+26}")
                        current_expected = "C"  # Next we expect C or C'
                    elif result == [1,0,0,1,0,1,1,1,0,0] or result == [1,1,1,1,0,0,1,1,0,0]:
                        syndrome = "C" if result == [1,0,0,1,0,1,1,1,0,0] else "C prime"
                        print(f"Syndrome is {syndrome} at position: {position+26}")
                        current_expected = "D"  # Next we expect D
                    elif result == [1,0,0,1,0,1,1,0,0,0]:
                        syndrome = "D"
                        print(f"Syndrome is D at position: {position+26}")
                        current_expected = "A"  # Next we expect A
                else:
                    # We're in the middle of a sequence - only look for the expected next syndrome
                    if current_expected == "B" and result == [1,1,1,1,0,1,0,1,0,0]:
                        syndrome = "B"
                        print(f"Syndrome is B at position: {position+26}")
                        current_expected = "C"
                    elif current_expected == "C" and (result == [1,0,0,1,0,1,1,1,0,0] or result == [1,1,1,1,0,0,1,1,0,0]):
                        syndrome = "C" if result == [1,0,0,1,0,1,1,1,0,0] else "C prime"
                        print(f"Syndrome is {syndrome} at position: {position+26}")
                        current_expected = "D"
                    elif current_expected == "D" and result == [1,0,0,1,0,1,1,0,0,0]:
                        syndrome = "D"
                        print(f"Syndrome is D at position: {position+26}")
                        current_expected = "A"
                    elif current_expected == "A" and result == [1,1,1,1,0,1,1,0,0,0]:
                        syndrome = "A"
                        print(f"Syndrome is A at position: {position+26}")
                        current_expected = "B"

                if syndrome:
                    # When a syndrome is detected:
                    if syndrome == "A":
                        # Insert Block 1 (bits 0-25)
                        current_group[0:26] = diff_stream[position:position+26]
                    elif syndrome == "B":
                        # Insert Block 2 (bits 26-51)
                        current_group[26:52] = diff_stream[position:position+26]
                    elif (syndrome == "C" or syndrome == "C prime"):
                        # Insert Block 3 (bits 52-77)
                        current_group[52:78] = diff_stream[position:position+26]
                    elif syndrome == "D":
                        # Insert Block 4 (bits 78-103)
                        current_group[78:104] = diff_stream[position:position+26]
                        
                        # Check if all blocks are filled
                        if None not in current_group:
                            # Decode the group (bitwise)
                            group_info = decode_group_bitwise(current_group)
                            if group_info:
                                print("\n=== RDS Group Decoded ===")
                                print(f"PI Code: {group_info['pi_code']:04X}")
                                print(f"Program Type: {group_info['pty_code']}")
                                if 'ps_name' in group_info:
                                    print(f"Program Service: {group_info['ps_name']}")
                                if 'radio_text' in group_info:
                                    print(f"Radio Text: {group_info['radio_text']}")
                            
                            # Reset for the next group
                            current_group = [None] * 104

                position += 1

                if position+26 > len(diff_stream):
                    sync = True
            
            
        block_count += 1
              
        
'''
fig, (phase) = plt.subplots(nrows=1)
fig.subplots_adjust(hspace=1.0)
phase.scatter(symbols_i_full, symbols_q_full, s=10)
#phase.scatter(ncoOut, ncoOut_q, s=10)
phase.set_ylim(-1.5, 1.5)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(i_samples, q_samples, s=10)
plt.title("IQ Constellation")
     

fig, (phase) = plt.subplots(nrows=1)
fig.subplots_adjust(hspace=1.0)
phase.scatter(symbols_i_full, symbols_q_full, s=10)
#phase.scatter(ncoOut, ncoOut_q, s=10)
phase.set_ylim(-1.5, 1.5)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(i_samples, q_samples, s=10)
plt.title("IQ Constellation")
        block_count += 1
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.show()
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.show()
'''