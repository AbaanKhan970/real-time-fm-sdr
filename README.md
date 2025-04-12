# 📻 Real-Time FM Receiver: SDR for Mono/Stereo Audio and RDS

This project implements a real-time software-defined radio (SDR) system capable of demodulating FM mono audio, stereo audio, and Radio Data System (RDS) signals using low-cost RF hardware and a Raspberry Pi 4.

The system processes 2.4 MS/s I/Q input streams and supports multiple modes with custom sample rates. Signal processing is implemented in Python for modeling and C++ for real-time execution, leveraging efficient filtering, resampling, multithreading, and Unix-style piping (`stdin/stdout`).

---

## 🚀 Features

- **Real-Time FM Demodulation** – Mono, stereo, and RDS audio/data decoding.
- **DSP Optimization** – Combined upsampling, filtering, and downsampling into a single resampler.
- **Stereo Carrier Recovery** – Implemented digital PLL for 38 kHz subcarrier regeneration.
- **RDS Decoding** – Extracted PI/PTY/PS fields using Manchester decoding and bit-wise parsing.
- **Multithreaded Architecture** – Separate threads for RF frontend , nono/stereo audio output and RDS data output with safe queues, mutexes and condition variables.
- **Unix Piping** – Designed as a streaming pipeline using `stdin`/`stdout` for block-wise processing.

---

## 🧠 Technologies Used

- **Languages**: C++, Python
- **Libraries**: NumPy, SciPy, Matplotlib
- **Hardware**: RTL-SDR USB dongle, Raspberry Pi 4
- **Concurrency, Multithreading**: `std::thread`, mutexes, condition variables
- **Signal Processing**: FIR filters, PLL, resamplers, convolution, Manchester decoding

---

### Python Modeling
Used the Python scripts (`fmMonoBlock.py`, `fmStereoBlock.py`, `fmRDS.py`) to prototype and visualize signal flow so that correctness can be verified before real-time implementation.

### Real-time implementation in C++
The real-time implementation of the project was done on a Rasberry-Pi 4 . The implementation was done in the file (`project.cpp`) and all the supporting functions such as the resampler are in (`filter.cpp`). The project switches between modes of mono audio and stereo audio using a Unix-pipelined style architecture . Additonally, multithreading was used to seperate the threads for RF front end, audio output (mono, stereo)
and RDS data.
