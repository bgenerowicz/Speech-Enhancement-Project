clear vars
%

[signal_clean, fs]=audioread('clean.wav');
[signal, fs]=audioread('clean+20n.wav');
[signal_beforedd, fs]=audioread('beforedd_mmse.wav');
[signal_afterdd, fs]=audioread('afterdd_mmse.wav');

win_time = 0.020 %seconds

win_len=ceil(win_time*fs)

STOI_error_orig = stoi(signal_clean, signal, fs);

STOI_error_before = stoi(signal_clean, signal_beforedd, fs);
STOI_error_after= stoi(signal_clean, signal_afterdd, fs);
