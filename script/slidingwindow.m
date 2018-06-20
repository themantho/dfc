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

% Get number of time series
numParcels = size(data,2);

% Get number of windows
numWindows = floor((size(data,1)-W)/(W-T) + 1);

% Initialize correlation heatmap
fc_swindow = zeros(numWindows,numParcels,numParcels);

% Loop Correlation
for iter = 1:numWindows
    
    % Get first W time series
    ts = data(((iter-1)*(W-T)+1):((iter-1)*(W-T)+W),:);
    
    % Get Rep matrix
    rep_matr = repmat(win,numParcels,1)
    
    fc_swindow(iter,:,:) = corrcoef((ts.*rep_matr'));
end