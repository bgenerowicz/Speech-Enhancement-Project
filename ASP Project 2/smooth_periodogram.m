function [ yp_smooth ] = smooth_periodogram(yp, SMOOTHING_TYPE, alpha)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


% create Bartlett estimate from noisy estimation
yp_smooth = zeros(size(yp));
% PARAMETER FOR BARTLETT SMOOTHING
smooth_length = 5;
bartlett_buffer = [];
lastvalue = yp(:,1);
for k = 1:size(yp, 2)
    if(strcmp(SMOOTHING_TYPE, 'EXPONENTIAL'))
        yp_smooth(:,k) = alpha*lastvalue + (1-alpha)*yp(:,k);
        lastvalue = yp_smooth(:,k);
    else
        bartlett_buffer = [bartlett_buffer yp(:,k)];
        if size(bartlett_buffer, 2) > smooth_length
            bartlett_buffer = bartlett_buffer(:,2:end); end
        yp_smooth(:,k) = mean(bartlett_buffer, 2);  
    end
end

end

