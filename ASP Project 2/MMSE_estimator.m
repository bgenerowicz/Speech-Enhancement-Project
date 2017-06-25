function [ sigma_n_estimates, prob_H1_y_history , p_bar_history] = MMSE_estimator( yp_smooth, alpha_bar, alpha_n, P_H0, prior_SNR )
% estimate the noise power using the MMSE estimation 
    % prior speech and noise probabilities
    P_H1 = 1 - P_H0;
    % prior SNR estimate:
    chi_H1 = 10^(prior_SNR/10);
    % initialization of the noise estimate 
    sigma_n = 0.3* mean(yp_smooth,2) * P_H0;
%     sigma_n = zeros(size(yp_smooth, 1), 1);
    % initial value for smoothing P_bar 
    P_bar = P_H1;
    % make a matrix to store the noise estimates in 
    sigma_n_estimates = zeros(size(yp_smooth));
    prob_H1_y_history = zeros(size(yp_smooth));
    p_bar_history = zeros(size(yp_smooth));
    for l = 1:size(yp_smooth,2)
        % compute the posterior probabilities
        prob_H1_y = 1 ./ (1 + (P_H0/P_H1)*(1+chi_H1)*exp( (-yp_smooth(:,l)./sigma_n) * (chi_H1)/(1+chi_H1)) );
        % P_bar is the exponentially smoothed version of prob_H1_y
        P_bar = alpha_bar*P_bar + (1-alpha_bar) * prob_H1_y;
        % update all the prob_H1_y values where the thhreshold for P_bar is
        % exceeded. This avoids stagnation [24]
        prob_H1_y(P_bar > 0.99) = min(0.99, prob_H1_y(P_bar > 0.99));
        prob_H0_y = 1 - prob_H1_y;
        est_noise = prob_H0_y.*yp_smooth(:,l) + prob_H1_y.*sigma_n;
        % exponential averaging filter 
        
        sigma_n = alpha_n*sigma_n + (1-alpha_n)*est_noise;
        sigma_n_estimates(:,l) = est_noise;
        prob_H1_y_history(:,l) = prob_H1_y;
        p_bar_history(:,l) = P_bar;
        
    end

end

