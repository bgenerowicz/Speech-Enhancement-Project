clear vars 
%  close all
%% Read the audio files: 
addpath('Files')
[speech1, fs]=audioread('speech16k.wav');
speech2=audioread('clean.wav'); ssnoise=audioread('SSN_Noise.wav'); nsnoise=audioread('n.wav');
noise2=audioread('intersection_soundjay.wav'); bnoise=audioread('babble-40talkers_fs16k.wav');
% determine length of speech audio 

N=length(speech1);
speech1=speech1(1:N,1);
win_time = 0.020 %seconds
win_len=ceil(win_time*fs)
win_len = win_len - mod(win_len, 2);
% mix audio signals together 
speech_ratio = 0.92;
noise_ratio = abs((1-speech_ratio) / 3);
% noise_ratio = 0.1;
noise_data = noise_ratio*0.1*randn(N, 1);
% noise_data = noise_ratio*noise2(1:N)...
%     +noise_ratio*bnoise(1:N)+...
%     noise_ratio*nsnoise(1:N);
speech_data = speech_ratio*speech1;
speech_data = speech_data(1:ceil(length(speech_data)/5));
noise_data = noise_data(1:ceil(length(noise_data)/5));

sumsound= speech_data + noise_data;
audiowrite('initial.wav',sumsound,fs) 

% find the true SNR 
speech_power = sum(sumsound .* sumsound);
noise_power = sum(noise_data .* noise_data);
true_SNR = speech_power / noise_power; 
true_SNR_db = 10*log10(true_SNR);
display(true_SNR_db);

%% PARAMETERS 
noise_estimation_type = 'MMSE';
% noise_estimation_type = 'MS';
% noise_estimation_type = 'test';
periodogram_smooth_type = 'EXPONENTIAL';
% speech_estimtion_type = 'ML';
speech_estimation_type = 'DD';
alpha_MMSE = 0.95;
alpha_n_MMSE=0.99;
alpha_periodogram = 0.85;
alpha_periodogram_noise = 0.85;
P_H0_MMSE = 0.5;
prior_SNR_MMSE = 15;  %prior average SNR for speech
hist_win = 130;
alphadd=0.2;




%% Segment audio and compute STFT
dataframes = buffer(sumsound, win_len, win_len/2);
noiseframes = buffer(noise_data, win_len, win_len/2);
win = Modhanning(win_len);
% apply hamming window 
dataframes = dataframes .* repmat(win, 1, size(dataframes,2));
noiseframes = noiseframes .* repmat(win, 1, size(noiseframes,2));
% compute FFT 
data_fft = fft(dataframes, win_len);
noise_fft = fft(noiseframes, win_len);
% take only first part of the data 
data_fft = data_fft(1:win_len/2+1, :);
noise_fft = noise_fft(1:win_len/2+1, :);


%% Compute power and smooth
% calculate the power of the signal and make some plots
yp = data_fft .* conj(data_fft);
y_noise = noise_fft .* conj(noise_fft);
figure(1); clf; subplot(221);
climits = [min(log(abs(data_fft(:)))) max(log(abs(data_fft(:))))];
imagesc(log(abs(data_fft)), climits);
title('Magnitude plot of audio FFT');
subplot(222);
imagesc(log(y_noise), climits);
title('Magnitude plot of noise FFT');
yp_smooth = smooth_periodogram(yp, periodogram_smooth_type, alpha_periodogram);
% yp_smooth =yp;
y_noise_smooth = smooth_periodogram(y_noise, periodogram_smooth_type, alpha_periodogram_noise);
% plot smoothed versus non-smoothed
figure(1); subplot(223); imagesc(log(yp_smooth), climits);
title('Smoothed Periodogram');
bands = [4 10 20 40];
figure(2); clf;
for k = 1:4
    band = bands(k);
    subplot(2, 2, k);
    
    xaxis = (1:size(yp_smooth, 2)) * (win_len / 2) / fs;
    plot(xaxis, 10*log10(yp_smooth(band, :))); hold on;
    plot(xaxis, 10*log10(yp(band,:)));
    legend('Smoothed','Not Smoothed');
    title(sprintf('band nr. %d', band));
%     title('Smoothed vs non-smoothed periodogram estimate');
end

%% Estimation of noise power and plot 
% MMSE estimation
[ sigma_n_estimates_MMSE, prob_H1_y_history, p_bar_history] = MMSE_estimator( yp, alpha_MMSE, alpha_n_MMSE, P_H0_MMSE, prior_SNR_MMSE);
% Minimum Statistics estimation
[ sigma_n_estimates_MS ] = MS_estimator( yp_smooth, hist_win );
if(strcmp(noise_estimation_type, 'MS'))
    sigma_n_estimates = sigma_n_estimates_MS;
elseif(strcmp(noise_estimation_type, 'MMSE'))
    sigma_n_estimates = sigma_n_estimates_MMSE;
else
    sigma_n_estimates=y_noise;
end

% pick a frequency band to display 
figure(3); clf; 
% subplot(311);
for k = 1:4
    subplot(2, 2, k);
    band = bands(k);
    xaxis = (1:size(yp_smooth, 2)) * (win_len / 2) / fs;
    plot(xaxis, 10*log10(yp_smooth(band, :))); hold on;
    plot(xaxis, 10*log10(sigma_n_estimates_MS(band,:)));
    plot(xaxis, 10*log10(y_noise_smooth(band,:)));
    title(sprintf('band nr. %d', band));
    plot(xaxis, 10*log10(sigma_n_estimates_MMSE(band, :)));
%     plot(xaxis, 10*log10(sigma_n_estimates(band, :)));
end

xlabel('Time (seconds)');
xlim([0 max(xaxis)]);
% xlim([0 1.2]);
legend('Smooth spectrum', 'Minimum Statistics Estimate', 'Noise level', 'MMSE estimate');
% 
% subplot(312); 
% plot(xaxis, prob_H1_y_history(band,:));
% xlim([0 max(xaxis)]);
% % xlim([0 1.2]);
% 
% subplot(313); 
% plot(xaxis, p_bar_history(band,:));
% xlim([0 max(xaxis)]);
% % xlim([0 1.2]);



%% Estimate speech power and applay wiener gain 

sigma_s_estimates = estimate_speech(yp_smooth, sigma_n_estimates, speech_estimation_type, alphadd);
% wiener gain: ratio signal power and power in y [ slide 47 estimation lecture]

gain_function = max( (sigma_s_estimates) ./(sigma_s_estimates+sigma_n_estimates), eps);
data_fft = data_fft.*gain_function;
% plot clean fft
figure(1); subplot(224);
imagesc(log(abs(data_fft)), climits);
title('Noise removed');

%% Reconstruct audio data 
% reconstruct data frames
reconstructed_dataframes = ifft(data_fft, win_len, 1);
reconstructed_audio = zeros((length(reconstructed_dataframes)+1)*win_len/2, 1);
idx = 2;
for k = 1:size(reconstructed_dataframes,2)
    reconstructed_audio(idx:idx+win_len-1) = ...
        reconstructed_audio(idx:idx+win_len-1) + reconstructed_dataframes(:,k);
    idx = idx + win_len/2;
end
reconstructed_audio = real(reconstructed_audio);
reconstructed_audio = reconstructed_audio(1:length(sumsound));
% play file and write to disk 
a = audioplayer(reconstructed_audio, fs);
a.play

% scales=[max(sumsound)-min(sumsound); max(reconstructed_audio)-min(reconstructed_audio)];
% scaling_factor=scales(1)/scales(2);
% reconstructed_audio=reconstructed_audio.*scaling_factor;
audiowrite('minimum_statistics_1.wav',reconstructed_audio,fs) 

%% Test the performance of the noise reduction 
MSE_error_orig = sum((sumsound - speech_data).^2);
STOI_error_orig = stoi(speech_data, sumsound, fs);
SEG_SNR_error_orig = seg_snr(sumsound, speech_data, win_len);

MSE_error_reduc = sum((reconstructed_audio - speech_data).^2);
STOI_error_reduc = stoi(speech_data, reconstructed_audio, fs);
SEG_SNR_error_reduc = seg_snr(reconstructed_audio, speech_data, win_len);


disp('MSE before:');
disp(MSE_error_orig);
disp('MSE after');
disp(MSE_error_reduc);

disp('STOI before:');
disp(STOI_error_orig);
disp('STOI after');
disp(STOI_error_reduc);

disp('SEG_SNR before:');
disp(SEG_SNR_error_orig);
disp('SEG_SNR after');
disp(SEG_SNR_error_reduc);

figure(4); clf;
subplot(411); 
lims = [min(sumsound) max(sumsound)];
plot(sumsound);
ylim(lims);
title('Noisy source');
subplot(412);
plot(speech_data);
ylim(lims);
title('Speech data');
subplot(413);
plot(reconstructed_audio);
ylim(lims);
title('Reconstructed audio');

subplot(414);
plot((reconstructed_audio-speech_data).^2)








