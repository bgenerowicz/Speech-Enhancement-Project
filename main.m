clear vars
%

[signal_clean, fs]=audioread('clean.wav');
[signal, fs]=audioread('clean+20n.wav');

win_time = 0.016 %seconds

win_len=ceil(win_time*fs)

STOI_error_orig = stoi(signal_clean, signal, fs);
