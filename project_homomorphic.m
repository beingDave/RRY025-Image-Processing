%% Clear All
clear all; clc; clf; close all;

%% Load Image := I
I = load('forest.mat');
I = im2double(I.('forestgray'));

%% Set Parameters := gammaH = 1.05; gammaL = 0.75; c = 1; D0 = 6;

% constant parameters
gammaH = 1.05;
gammaL = 0.75;
D0 = 60;
A = 1e-10;
c = 1;

fprintf('Varying gammaH\n');
gammaH = [1.00 1.25 1.75];
compute(gammaH, gammaL, D0, A, c, I, 'varyGH'); gammaH = 1.05;

fprintf('Varying gammaL\n');
gammaL = [0.15 0.50 0.80 0.95];
compute(gammaH, gammaL, D0, A, c, I, 'varyGL'); gammaL = 0.75;

fprintf('Varying sqrt(c)/D0\n');
D0 = [0.60 4.50 6.00 45.0];
compute(gammaH, gammaL, D0, A, c, I, 'varyD0'); D0 = 6;

fprintf('Best Image\n');
gammaH = 1.05;
gammaL = 0.75;
D0 = 60;
compute(gammaH, gammaL, D0, A, c, I, 'best');

%% Compute Function: Varies Parameters, Plots
function compute(gammaH, gammaL, D0, A, c, I, text)
    for i = 1:length(gammaH)
        for j = 1:length(gammaL)
            for k = 1:length(D0)
                I_op = homomorphic(gammaH(i), gammaL(j), k, A, c, I); imshow(I_op);
                saveas(gcf, sprintf('%s_gH_%.0f_gL_%.0f_D0_%.1f.png', text, gammaH(i)*1000, gammaL(j)*1000, D0(k)), 'png')
                %imwrite(I_op,sprintf('%s_gH_%.0f_gL_%.0f_D0_%d.png', text, gammaH(i)*1000, gammaL(j)*1000, D0(k)));
            end
        end
    end
end

%% Homomorphic Filtering Function
function [I_op, H] = homomorphic(gammaH, gammaL, D0, A, c, I_in)
    %% Image Transformation : I_temp = Shift(FFT2 { Zero-Padding(log(A + I_in)) } )
    
    % get I [rows, columns]
    [M, N] = size(I_in);
    
    % compute nearest minimum power of 2 for zero-padded I
    P = 2^(nextpow2(2*M-1));                             
    Q = 2^(nextpow2(2*N-1));                          
    
    % zero padding I_in
    I_temp = padarray(log(A + I_in), [P-M Q-N], 0, 'post');
    % fprintf('Zero-Padded Image Dimensions: %d x %d\n', size(I_temp,1), size(I_temp,2));

    % transform
    I_temp = fftshift(fft2(I_temp));
    
    %% Generate Filter: H(u,v)
    D = @(u, v) ((u-ceil(P/2)).^2 + (v-ceil(Q/2)).^2);
    H = @(u, v) (gammaH - gammaL)*(1 - exp((-c*D(u,v))/(D0^2))) + gammaL;
    H = H((1:P)', (1:Q));
    
    %% Revert Image Transformation: I_op = [ exp (Real { Remove Zero-Padding(Inv_FFT2(Inv_Shift(H*I_temp))) } ) - A]
    
    % inverse fft
    I_op = ifft2(ifftshift(H.*I_temp));
    
    % remove padding
    I_op = I_op(1:M, 1:N);
    
    % revert transformation
    I_op = exp(real(I_op)) - A;

    %fprintf("max: %.5f\tmin: %.5f\tgammaL: %.2f\tgammaH: %.2f\tD0: %.2f\n", max(max(I_op)), min(min(I_op)), gammaL, gammaH, D0);
    
end