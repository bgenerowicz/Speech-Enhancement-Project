function [ sigma_n_estimates ] = MS_estimator( yp_smooth, hist_win )
% estimate the noise power using minimum statistics
    % estimate the minimum power 
    sigma_n_estimates = zeros(size(yp_smooth));
    pastvalues = [];
    for k = 1:size(yp_smooth, 2)
        % add a new element to the end of the sliding window
        pastvalues = [pastvalues yp_smooth(:,k)];
        if size(pastvalues, 2) > hist_win
            % remove oldest element from start of sliding window
            pastvalues = pastvalues(:, 2:end);
        end
        % compute the new minimum for all bands-
        sigma_n_estimates(:,k) = min(pastvalues, [], 2);
    end
end

