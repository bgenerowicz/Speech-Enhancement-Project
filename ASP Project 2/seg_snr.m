function [ tot_seg_snr ] = seg_snr( x, y, win_len )
% computes the segmented SNR of the estimated signal x with the reference
% signal y. win_len: segment length

N = length(y);
if( length(x) ~= N)
    error('Vectors must be same length');
end

%x = x(y ~= 0);
%y = y(y ~= 0);

% segment the data 
x_seg = buffer(x, win_len);
y_seg = buffer(y, win_len);
% find number of frames 
nFrames = length(y_seg);

eps =1e-10;
% compute signal power 
y_seg_power = ones(1, win_len) * y_seg.^2;
% compute square error 
err_seg_power = ones(1, win_len) *  (x_seg-y_seg).^2 + eps;

% find the SNR in dB per segment
snr_seg = 10*log10( y_seg_power(y_seg_power~=0) ./ err_seg_power(y_seg_power~=0) ); 
% find the average SNR per segment
tot_seg_snr = sum(snr_seg) / nFrames;

plot(snr_seg)
end

