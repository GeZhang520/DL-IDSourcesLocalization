function [rx_noise, distribution, R_free, R_noise] = ...
    IDSourceGenerator(array, nsample, ang, spread, snr, ang_range, stype)
% Generate ID source array receive signal
%   array£ºArray element position
%   nsample£ºsnapshot
%   ang£ºNominal DOA labels
%   spread£ºAngular spread labels
%   snr
%   ang_range£ºArray observation range
% 
% return£º
%   rx_noise£ºReceived signal
%   distribution£ºID source angular distribution
%   R_free£ºNoise-free received signal covariance matrix
%   R_noise: Noisy received signal covariance matrix

sqrt_3 = 1.7321;
sqrt_0_5 = 0.7071;

mu = ang;
sqrt_sigma =  spread;

naz = length(ang_range);

signal_num = length(mu);
switch signal_num
    case 1
        if stype==1
            % Gaussian distribution 
            distribution = normpdf(ang_range, mu, sqrt_sigma);
        else
            % uniform distribution
            a = mu - sqrt_3 * sqrt_sigma;
            b = 2*mu - a;
            distribution = unifpdf(ang_range, a, b);
        end
    case 2
        a = mu - sqrt_3 * sqrt_sigma;
        b = 2*mu - a;
        if stype(1)==1
            distribution(1, :) = normpdf(ang_range, mu(1), sqrt_sigma(1));
        else
            distribution(1, :) = unifpdf(ang_range, a(1), b(1));   
        end
        if stype(2)==1
            distribution(2, :) = normpdf(ang_range, mu(2), sqrt_sigma(2));
        else
            distribution(2, :) = unifpdf(ang_range, a(2), b(2));
        end
        distribution = (distribution(1, :) + distribution(2, :));
        distribution = distribution./sum(distribution);
    otherwise
        error('Too many signals\n');
end

x = sqrt_0_5 * ( randn(naz, nsample) + 1i*randn(naz, nsample) );

derad = pi/180;
% Calculate array steering vector
A = exp((-1i*2*pi*array.') * sin(ang_range*derad));

% Calculation of Noise-free received signal
rx_free = A * (sqrt(diag(distribution))* x);

% Add Gaussian noise to the received signal
rx_noise = awgn(rx_free, snr, 'measured', 'db');

% Calculate the covariance matrix
R_noise = rx_noise*rx_noise'/nsample;
R_free = rx_free*rx_free'/nsample;

end
