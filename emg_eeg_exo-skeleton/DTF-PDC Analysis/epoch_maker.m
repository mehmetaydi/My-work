

function [T,filtered_signal] = epochh(x,movement_points, fs,before,after);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  [T,filtered_signal] = epochh(x,movement_points, fs,before,after)                     %
%% This function split the signal into epochs                                            %
%%                                                                                       %
%%  USAGE:                                                                               %
%%        [T,filtered_signal] = epochh(x,movement_points, fs,before,after)               %
%%              Inputs:                                                                  %
%%                   x : Signal File (easy_file)                                         %
%%                   movement_points : movement intention where the subject moves        %
%%                   fs : Sampling Frequency                                             %
%%                   before : the time before movement in second                         %
%%                   after : the time after movement in second                           %
%%              Outputs:                                                                 %
%%                       T : Preprocessed signal                                         %
%%                  Output shape [N_epochs L_epoch CH]                                   %
%% N_epochs = the number of epochs                                                       %
%% L_epoch = the length of each epoch in samples                                         %
%% CH =number of channel(s)                                                              %                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(x(:,1))
    xd = decimate(x(i,:), 5,'fir');

    % high pass filtering
    [n,Wn] = buttord([0.1]/50, [0.05]/50, 3, 20);
    [b,a]=butter(n, Wn, 'high');

    filtsig=filtfilt(b,a,xd);  %filtered signal
    filtsig =filtsig-mean(filtsig);
%     filtsig = smoothdata(filtsig);
%     filtsig =(filtsig)/max(filtsig);
    filtered_signal(:,i) =filtsig;
    for j =1:length(movement_points)
    segment = filtsig((movement_points(j)-fs*before):(movement_points(j)+fs*after-1));
%     segment = segment ./ max(abs(segment)) ; 
    
    seg(j,:) =segment;
    end
    T(:,:,i) =seg;
end
