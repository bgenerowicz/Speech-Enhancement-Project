%% Read the audio files:
[speech1, fs]=audioread('speech16k.wav');
speech2=audioread('clean.wav'); ssnoise=audioread('SSN_Noise.wav'); nsnoise=audioread('n.wav');
noise2=audioread('intersection_soundjay.wav'); bnoise=audioread('babble-40talkers_fs16k.wav');
% load dpfile.mat
% determine length of speech audio
N=length(speech1);
win_time = 0.02 %seconds
win_len=ceil(win_time*fs)
% mix audio signals together
speech_ratio = 0.90;
noise_ratio = (1-speech_ratio) / 3;
sumsound=speech_ratio*speech1+noise_ratio*ssnoise(1:N)+...
    noise_ratio*nsnoise(1:N)+noise_ratio*bnoise(1:N);
filename = sprintf('initial.wav');
audiowrite(filename,sumsound,fs);
sumnoise=noise_ratio*ssnoise(1:N)+noise_ratio*nsnoise(1:N)+noise_ratio*bnoise(1:N);
filename = sprintf('initialnoise.wav');
audiowrite(filename,sumnoise,fs);
% % boost volume a little bit
% sumsound= sumsound;
%% Segment audio and compute STFT
dataframes = buffer(sumsound, win_len*2, win_len);
win = sqrt(hamming(win_len*2));
% apply hamming window
dataframes = dataframes .* repmat(win, 1, length(dataframes));

data_fft = fft(dataframes, win_len*2);

data_fft= data_fft(1:fix(win_len)+1,:);

%% noise PSD estimator
yp=data_fft.*conj(data_fft);
% 
% load('yp.mat')
% yp=yf';
% yp(end,:)=[];
% yp=[yp zeros(320,2)];
[nr,nrf]=size(data_fft); %nr=320 nrf=1681
asnr=15; %active SNR in dB
xi_h1=10^(asnr/10); % speech-present SNR
xi=xi_h1
ssnr1=1/(1+xi_h1)-1;
pspri=0.5; %prior speech probability
psnoi=0.5;%prior speech probability
psini=0.5 ;% initial speech probability [0.5]
pfac=(psnoi/pspri)*(1+xi_h1); %pnoise/pspeech
pslp=repmat(psini,nr,1); % initialize smoothed speech presence prob
sigma_n=psini*mean(yp(:,1:10),2); %initial noise estimate
% x=zeros(nr,nrf);
ddalpha=0.1
nu=0.4
% loop through all frequency bands 
for l=1:nrf
    % extract a single frequency band 
    ypl=yp(:,l);
    % probability of speech presence H1 given y
    ph1y=(1+pfac*exp(-ypl./sigma_n*(xi_h1/(1+xi_h1)))).^(-1);
    % exponential alpha-smoothing of speech probability with alpha = 0.1
    pslp=0.9*pslp+0.1*ph1y;
    ph1y=min(ph1y,1-0.01*(pslp>0.99)); % limit ph1y (24) % limit ph1y (24)
    xtr=(1-ph1y).*ypl+ph1y.*sigma_n; % estimated raw noise spectrum (22)
    sigma_n=0.8*sigma_n+0.2*xtr;  % smooth the noise estimate (8)
    sigma_nf(:,l)=sigma_n;
%     zeta=ypf./sigma_nf;
%     
%     Ea=pA (xi, nu, zeta, abs(ypl));
%     xi=ddalpha*Ea./sum(sigma_n)+1-ddalpha*max(ypl-1,0)/sum(sigma_n)
end
alow=-5;       % SNR for maximum a (set to Inf for fixed a)
ahigh=50;       % SNR for minimum a

ypf=sum(yp,1);
dpf=sum(sigma_nf,1);
%Decision directed approach




v=sqrt(sigma_nf./yp);

data_fft=data_fft.*(1-v);



%% speech PSD estimator





%% reconstruct data frames
reconstructed_dataframes=real(ifft([data_fft;(flipud(conj(data_fft(2:end-1,:))))],win_len*2,1)); 
reconstructed_dataframes =reconstructed_dataframes;
% reconstructed_dataframes = ifft(data_fft, win_len*2, 1) ;

% reconstructed_noise = ifft(x, win_len, 1);
% reconstructed_dataframes=reconstructed_dataframes-reconstructed_noise;
%% reconstruct segments
reconstructed_audio = zeros((length(reconstructed_dataframes)+1)*win_len, 1);
idx = 1;
for k = 1:length(reconstructed_dataframes)
    reconstructed_audio(idx:idx+win_len*2-1) = ...
        reconstructed_audio(idx:idx+win_len*2-1) + reconstructed_dataframes(:,k);
    idx = idx + win_len;
end
% a = audioplayer(reconstructed_audio, fs);



%% gain function



%% multi-microphone system

filename = sprintf('final.wav');
audiowrite(filename,reconstructed_audio,fs);