clear vars
%

[signal_clean, fs]=audioread('clean.wav');
[signal, fs]=audioread('clean+20n.wav');
[signal_beforedd_min, fs]=audioread('reconstruction_beforedd_min.wav');
[signal_afterdd_min, fs]=audioread('reconstruction_afterdd_min.wav');
[signal_beforedd_mmse, fs]=audioread('reconstruction_beforedd_mmse.wav');
[signal_afterdd_mmse, fs]=audioread('reconstruction_afterdd_mmse.wav');
[signal_beforedd_mmse_ns, fs]=audioread('reconstruction_beforedd_mmse_ns.wav');
[signal_afterdd_mmse_ns, fs]=audioread('reconstruction_afterdd_mmse_ns.wav');
[signal_beforedd_mmse_ns_nn, fs]=audioread('reconstruction_beforedd_mmse_ns_nn.wav');
[signal_afterdd_mmse_ns_nn, fs]=audioread('reconstruction_afterdd_mmse_ns_nn.wav');
[signal_clean_n, fs]=audioread('clean_n.wav');
[signal_beforedd_min_i, fs]=audioread('reconstruction_intersect_before_min.wav');
[signal_afterdd_min_i, fs]=audioread('reconstruction_intersect_after_min.wav');
[signal_beforedd_mmse_i, fs]=audioread('reconstruction_intersect_before_mmse.wav');
[signal_afterdd_mmse_i, fs]=audioread('reconstruction_intersect_after_min.wav');

win_time = 0.020 %seconds

win_len=ceil(win_time*fs)

STOI_error_orig = stoi(signal_clean, signal, fs);
STOI_error_before_min = stoi(signal_clean, signal_beforedd_min, fs);
STOI_error_after_min= stoi(signal_clean, signal_afterdd_min, fs);

STOI_error_before_mmse = stoi(signal_clean, signal_beforedd_mmse, fs);
STOI_error_after_mmse = stoi(signal_clean, signal_afterdd_mmse, fs);
STOI_error_before_mmse_ns = stoi(signal_clean, signal_beforedd_mmse_ns, fs);
STOI_error_after_mmse_ns = stoi(signal_clean, signal_afterdd_mmse_ns, fs);

STOI_error_before_mmse_ns_nn = stoi(signal_clean_n, signal_beforedd_mmse_ns_nn, fs);
STOI_error_after_mmse_ns_nn = stoi(signal_clean_n, signal_afterdd_mmse_ns_nn, fs);

STOI_error_before_mmse_i= stoi(signal_clean, signal_beforedd_mmse_i, fs);
STOI_error_after_mmse_i = stoi(signal_clean, signal_afterdd_mmse_i, fs);

STOI_error_before_min_i= stoi(signal_clean, signal_beforedd_min_i, fs);
STOI_error_after_min_i = stoi(signal_clean, signal_afterdd_min_i, fs);