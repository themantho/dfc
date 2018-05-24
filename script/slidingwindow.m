function fc_swindow = slidingwindow(data,W,T,winType)

% This function computes the matrix for the sliding window analysis for
% dynamic functional connectivity.
%
% A temporal window (W) is chosen, and within the temporal interval (t=1 to t=w_len), 
% Pearson's correlation coefficient is computed between each pair of time courses.
% Then the window is shifted by a step T, and the same calculations are repeated over the time interval [1+ T, W + T]. 
% This process is iterated until the window spans the end part of the timecourses, to eventually obtain a connectivity timecourse.

% INPUTS:
% data                 : input timeseries of size m x n, where m is the number of timeseries and n is the length of each timeseries in samples
% W                    : length of each sliding window in samples
% T                      : overlap between successive sliding windows in samples
% winType          : if specified as 'gauss', will implement a gaussian window, else will implement a rectangular window.

% OUTPUT
% fc_swindow    : matrix of size m x m x p where m is the number of timeseries and p is the number of sliding windows.

if (strcmp(winType, 'gauss'))
    win = gausswin(W)'; % gaussian window
else
    win = ones(1,W); % rectangular window
end

numParcels = size(data,1);

numWindows = floor((size(data,2)-W)/(W-T) + 1);
fc_swindow = zeros(numParcels,numParcels,numWindows);
for iter = 1:numWindows
    ts = data(:,...
        ((iter-1)*(W-T)+1):((iter-1)*(W-T)+W));
    fc_swindow(:,:,iter) = corrcoef((ts.*repmat(win,numParcels,1))');
end