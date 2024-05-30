Nt = 4;          % Number of transmit antennas
Nr = 4;          % Number of receive antennas
Nc = 64;         % Number of subcarriers
Nsyms = 10;      % Number of OFDM symbols
SNRdB_values = 0:5:30; % Range of SNR values in dB
N_realizations = 100; % Number of Monte Carlo realizations

% Preallocate for storing NMSE values
nmse_values = zeros(size(SNRdB_values));

% Loop over different SNR values
for idx = 1:length(SNRdB_values)
    SNRdB = SNRdB_values(idx);
    SNR = 10^(SNRdB/10);
    
    mse_temp = 0; 

    % Loop over Monte Carlo realizations
    for realization = 1:N_realizations
        % Generate MIMO channel 
        H = (randn(Nr, Nt, Nc) + 1j*randn(Nr, Nt, Nc)) / sqrt(2); % Normalized channel

        % Generate random transmit symbols (QPSK in this example)
        X = 2*(randi([0 1],Nt,Nsyms*Nc)-0.5) + 1j*2*(randi([0 1],Nt,Nsyms*Nc)-0.5);
        X = reshape(X, Nt, Nc, Nsyms);

        % Generate received signal (including noise)
        Y = zeros(Nr, Nc, Nsyms);
        noisePower = mean(abs(H(:)).^2) / SNR;
        for k = 1:Nc
            for l = 1:Nsyms
                Y(:, k, l) = H(:, :, k) * X(:, k, l) + sqrt(noisePower / 2) * (randn(Nr, 1) + 1j * randn(Nr, 1));
            end
        end

        % Reshape Y properly for Tucker Decomposition
        Y_tensor = tensor(Y);

        % SALSA Algorithm (Tucker Decomposition Implementation)
        % Set ranks for Tucker decomposition based on the dimensions of H
        R = [min(Nr, size(Y, 1)), min(Nt, size(Y, 2)), min(Nc, size(Y, 3))];
        maxIter = 5; % Maximum number of iterations

        % Perform Tucker decomposition (assuming you have tucker_als function)
        est_factors = tucker_als(Y_tensor, R, 'tol', 1e-6, 'maxiters', maxIter); 

        % Channel Estimation (using estimated factors)
        U1 = est_factors.U{1};
        U2 = est_factors.U{2};
        U3 = est_factors.U{3};
        core_tensor = est_factors.core;

        % Estimate H from Tucker factors
        H_est_salsa = ttm(core_tensor, {U1, U2, U3});
        H_est_salsa = double(H_est_salsa); % Convert tensor to double

        % Reshape H_est_salsa to match the size of H along the last two dimensions
        if numel(H_est_salsa) == numel(H)
            H_est_salsa = reshape(H_est_salsa, size(H)); % Reshape to match H
        else
            % Adjust the reshaping to match the size of H along the last two dimensions
            H_est_salsa = reshape(H_est_salsa, size(H)); % Reshape to match H
        end

        % Performance Evaluation (accumulate MSE)
        mse_temp = mse_temp + norm(H(:) - H_est_salsa(:), 'fro')^2; 
    end

    % Calculate and store average NMSE
    nmse_values(idx) = mse_temp / (N_realizations * norm(H(:))^2); 
end

% Plot NMSE vs. SNR 
figure;
semilogy(SNRdB_values, nmse_values, '-o', 'LineWidth', 1.5);
xlabel('SNR (dB)');
ylabel('NMSE');
title('SALSA MIMO-OFDM Channel Estimation Performance');
grid on;
