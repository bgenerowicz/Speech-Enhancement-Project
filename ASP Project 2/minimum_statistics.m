clear vars 
% close all
%% Read the audio files: 
[speech1, fs]=audioread('speech16k.wav');
speech2=audioread('clean.wav'); ssnoise=audioread('SSN_Noise.wav'); nsnoise=audioread('n.wav');
noise2=audioread('intersection_soundjay.wav'); bnoise=audioread('babble-40talkers_fs16k.wav');
% determine length of speech audio 
N=length(speech1);
win_time = 0.020 %seconds
win_len=ceil(win_time*fs)
% mix audio signals together 
speech_ratio = 2;
noise_ratio = abs((1-speech_ratio) / 3);
noise_ratio = 0.1;
sumsound=speech_ratio*speech1+noise_ratio*0.1*randn(N, 1);
% boost volume a little bit 
sumsound= sumsound;

sumsound = sumsound(1:ceil(length(sumsound)/8));
%% Segment audio and compute STFT
dataframes = buffer(sumsound, win_len, win_len/2);
win = hann(320);
% apply hamming window 
dataframes = dataframes .* repmat(win, 1, length(dataframes));

data_fft = fft(dataframes, win_len);
x = data_fft(:,100);




% take only first part of the data 
data_fft = data_fft(1:win_len/2+1, :);
figure(1);
imagesc(log(abs(data_fft)))
title('Only positive part of fft');

% calculate the power of the signal 
yp = data_fft .* conj(data_fft);
figure(2);
imagesc(log(yp));

% create Bartlett estimate from noisy estimation
yp_smooth = zeros(size(yp));
% PARAMETER FOR BARTLETT SMOOTHING
smooth_length = 5;
bartlett_buffer = [];
SMOOTHING_TYPE = 'MOVING_AVERAGE';
lastvalue = yp(:,1);
alpha = 0.95;
for k = 1:size(yp, 2)
    if(strcmp(SMOOTHING_TYPE, 'EXPONENTIAL'))
        yp_smooth(:,k) = alpha*lastvalue + (1-alpha)*yp(:,k);
        lastvalue = yp(:,k);
    else
        bartlett_buffer = [bartlett_buffer yp(:,k)];
        if size(bartlett_buffer, 2) > smooth_length
            bartlett_buffer = bartlett_buffer(:,2:end); end;
        yp_smooth(:,k) = mean(bartlett_buffer, 2);  
    end
end
figure(5); imagesc(log(yp_smooth));

% estimate the minimum power 
minimumlevel = zeros(size(yp_smooth));
pastvalues = [];
% PARAMETER FOR MINIMUM STATISTICS SMOOTHING
hist_win = 20;
for k = 1:size(yp_smooth, 2);
    % add a new element to the end of the sliding window
    pastvalues = [pastvalues yp_smooth(:,k)];
    if size(pastvalues, 2) > hist_win
        % remove oldest element from start of sliding window
        pastvalues = pastvalues(:, 2:end);
    end
    % compute the new minimum for all bands
    minimumlevel(:,k) = min(pastvalues, [], 2);
end
% pick a frequency band to display 
figure(3); clf;
band = 40;
plot(log(yp_smooth(band, :))); hold on;
plot(log(minimumlevel(band,:)));
plot(log(yp(band, :)));
legend('Smooth spectrum', 'Minimum power level', 'Unsmoothed spectrum');

% wiener gain: ratio signal power and power in y 
%[ slide 47 noise estimation lecture]
gain_function = max( (yp_smooth - minimumlevel) ./ yp_smooth, 0.2);

data_fft = data_fft.*gain_function;
        
figure(1);
imagesc(log(abs(data_fft)))
title('Noise removed');


data_fftt = data_fft - minimumlevel;






% make fft symmetric again 
data_fft = [data_fft; flipud(conj(data_fft(2:end-1,:)))];

% reconstruct data frames
reconstructed_dataframes = ifft(data_fft, win_len, 1);

%% reconstruct segments 
reconstructed_audio = zeros((length(reconstructed_dataframes)+1)*win_len/2, 1);
idx = 1;
for k = 1:length(reconstructed_dataframes);
    reconstructed_audio(idx:idx+win_len-1) = ...
        reconstructed_audio(idx:idx+win_len-1) + reconstructed_dataframes(:,k);
    idx = idx + win_len/2;
end

a = audioplayer(reconstructed_audio, fs);
a.play

audiowrite('minimum_statistics_1.wav',reconstructed_audio,fs);

