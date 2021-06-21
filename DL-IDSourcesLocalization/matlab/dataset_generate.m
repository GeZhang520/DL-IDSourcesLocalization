clc;
clear;
close all;

N          = 10;                   % Number of array elements                   
dd         = 0.5;                  % Array element spacing
array_pos  = 0:dd:(N-1)*dd;        % Array element position

resolution = 1;                    % Resolution of array observation
angs_range = [-80:resolution:80];  % Array observation range
nsamp      = 100;                  % snapshot
snr         = [0:4:20];             % dataset snr
max_snum   = 2;                    % Maximum number of sources


% -------------- generate train-dataset -------------
nangs = length(angs_range);
each_count = 5000;

signal_cov    = zeros(nangs*each_count, N, N, 2);
angle_labels  = zeros(nangs*each_count, max_snum);
spread_labels = zeros(nangs*each_count, max_snum);

all_count = nangs*each_count;
disp(all_count);

count = 1;
for k=1:nangs
    for w=1:each_count  
        amount = randperm(max_snum, 1);          % Number of sources
        stype  = randi(2, 1, amount);            % Distribution type
                                                 % 1: Gaussian distribution
                                                 % 2: uniform distribution
        ang    = sort(randperm(121, amount)-61); % Nominal DOA of the target
        if amount==2
            while ( abs(ang(1)-ang(2)) )<20
                ang = sort(randperm(121, amount)-61); 
            end
        end       
        spread = randperm(10, amount);           % Angular spread of the target
        db     = snr(randperm(length(snr), 1));  % Randomly selected SNR
        
        [~, ~, ~, signal_cov_complex] ...
            = IDSourceGenerator(array_pos, nsamp, ang, spread, db, angs_range, stype);

        real_s = real(signal_cov_complex);
        imag_s = imag(signal_cov_complex);
        mm = max(max(signal_cov_complex));
        
        signal_cov(count, :, :, 1) = real_s ./ mm;  % Normalization and 
        signal_cov(count, :, :, 2) = imag_s ./ mm;  % transformation of covariance matrix
        
        angle_labels(count, 1:amount) = ang;
        spread_labels(count, 1:amount) = spread;

        count = count+1;    
    end
    fprintf('%d of %d, %02.02f%%\n', count-1, all_count, (count-1)/all_count*100);
end
file_name = ['./dataset/train.mat'];
save(file_name, 'signal_cov', 'angle_labels', 'spread_labels');
% --------------------------------------------------------------

% -------------------- test -----------------------
test_count = 5;
signal_num = 2;
stype = [1, 2];
signal_cov_test = zeros(test_count, N, N, 2);
angle_labels_test = zeros(test_count, signal_num);
spread_labels_test = zeros(test_count, signal_num);
for w=1:test_count
    db = 10;
    ang = sort(randperm(121, signal_num)-61);
    spread = randperm(10, signal_num);
    
    [~, ~, ~, signal_cov_complex] ...
     = IDSourceGenerator(array_pos, nsamp, ang, spread, db, angs_range, stype);
    
    real_s = real(signal_cov_complex);
    imag_s = imag(signal_cov_complex);
    mm = max(max(signal_cov_complex));
    
    signal_cov_test(w, :, :, 1) = real_s ./ mm;
	signal_cov_test(w, :, :, 2) = imag_s ./ mm;
    
    angle_labels_test(w, 1:signal_num) = ang;
	spread_labels_test(w, 1:signal_num) = spread;
end
file_name = ['./dataset/test.mat'];
save(file_name, 'signal_cov_test', 'angle_labels_test', 'spread_labels_test', 'signal_num');
% -------------------------------------------------
