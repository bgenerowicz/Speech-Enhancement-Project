function [ sigma_s_estimates ] = estimate_speech( yp_smooth, sigma_n_estimates, estimation_type , alphadd)
% find the speech power estimate from the noise power and signal power
    if strcmp(estimation_type, 'ML')
        %% ML Smoothing
        [row,col]=size(yp_smooth);
        L=2;
        sigma_s_estimates=yp_smooth - sigma_n_estimates;
        for l = L+1:size(yp_smooth,1)
            sigma_s_estimates(:,l)=mean((yp_smooth(:,l-L:l)),2) - sigma_n_estimates(:,l);
        end
    else
        %% Decision Directed Approach
        sigma_s_estimates=yp_smooth - sigma_n_estimates;
        for l = 2:size(yp_smooth,2)
            sigma_s_estimates(:,l)=max(alphadd*(sigma_s_estimates(:,l-1)) + (1-alphadd)*(abs(yp_smooth(:,l)-sigma_n_estimates(:,l))), eps);
%             sigma_s_estimates(:,l)=alphadd*(sigma_s_estimates(:,l-1)./sigma_n_estimates(:,l))...
%                 + (1-alphadd)*max((yp_smooth(:,l)./sigma_n_estimates(:,l)-1), eps);
        end
    end
end

