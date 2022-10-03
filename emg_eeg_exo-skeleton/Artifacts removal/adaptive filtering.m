
%[hdr, data_eeg]  = edfread('20210224104858_test1.edf');
% clc
% clear all
data_eeg = importdata('easy_file_new2.txt');


%writetable(acceleration_test1_matlab);
ECG = data_eeg(20,:);
figure(1)

ECG = transpose(ECG);
EMG =  data_eeg(5,:);
EMG = transpose(EMG)-mean(EMG);

% [t, w] = adaptive_filter(EMG,ECG, 0.0006,5);
% writematrix(t,'cleanemg.txt');
[t, w] = adaptive_filter(EMG,ECG, 0.0000004,6);
writematrix(t,'cleanedEMG11.txt');
%t=t(0:10000)
subplot(211),plot(t);
% axis([0 63000 -300 300]);
subplot(212),plot(EMG'-mean(EMG));
% axis([0 63000 -300 300]);
%figure
%plot(w')
function [y, w] = adaptive_filter(x, ref, mu, M);

%x = [(zeros(M, 1) + x(1)); x];
ref = [(zeros(M, 1) + ref(1)); ref];
w = zeros(M, length(x));
y = zeros(1, length(x));
w(:,1) = zeros(M, 1) + 1/M;
for i = 1:length(x)
    n_e = w(:,i)' * flipud(ref(i:(i+M-1)));
    e_n = x(i) - n_e;
    y(i) = e_n;
    for j = 1:M
        w(j, i+1) = w(j, i) + mu * e_n * ref(i + M - j);        
    end
%    w(:, i+1) = w(:, i+1)./sum(w(:, i+1));
end
end
