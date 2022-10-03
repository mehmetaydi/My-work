clear
clc
close all
warning off

%% Time-invariant simulation for gOPDC analysis represented in ref [1] (Fig. 3)
%%% Written by: Amir Omidvarnia, 2013

%%% Ref [1]: A.  Omidvarnia,  G.  Azemi,  B.  Boashash,  J.  O.  Toole,  P.  Colditz,  and  S.  Vanhatalo, 
%%% “Measuring  time-varying  information  flow  in  scalp  EEG  signals:  orthogonalized  partial 
%%% directed coherence,”  IEEE  Transactions on Biomedical Engineering, 2013  [Epub ahead of print]

%% Time-invariant MVAR model 
Fs = 100; % Sampling frequency
Fmax = Fs/2; % Cut off frequency (Hz), should be smaller than Fs/2
Nf = 2*Fmax; 
% y = zeros(10000,5);
easy_file = importdata('20210414143751_Patient01.easy'); % easy_file_102517
easy_file = easy_file(9777:end,:);
easy_file =transpose(easy_file);
x =vertcat(easy_file(5,:), easy_file(16,:), easy_file(13,:), easy_file(15,:), easy_file(11,:), easy_file(19,:));

for i = 1:length(x(:,1))
    xd = decimate(x(i,:), 5,'fir');

    % high pass filtering
    [n,Wn] = buttord([0.1]/50, [0.05]/50, 3, 20);
    [b,a]=butter(n, Wn, 'high');

    filtsigg=filtfilt(b,a,xd);  %filtered signal
    filtsigg =filtsigg-mean(filtsigg);
    filtsigg = filtsigg(((606)-Fs*1):(606+Fs*1-1));
    y(:,i) =filtsigg
end
% y =transpose(y);
L = size(y,1); % Number of samples
CH = size(y,2); % Number of channels
a54 = zeros(1,L);
a45 = zeros(1,L);
%% 
%% Time-invariant MVAR parameter estimation
[w, A, C, sbc, fpe, th] = arfit(y, 1, 20, 'sbc'); % ---> ARFIT toolbox
[tmp,p_opt] = min(sbc); % Optimum order for the MVAR model

%% Connectivity measures (PDC, gOPDC etc)
[DTFs GPDC,OPDC,PDCs,GOPDC,S] = PDC_dDTF_imag(A,C,p_opt,Fs,Fmax,Nf);

%% Plot



%%%%
figure, % ---> gPDC
s1 = 0;
clear mask
GPDC = abs(GPDC);

x_max = Fmax;
y_max = .8;

for i = 1 : CH
    for j = 1 : CH
        s1 = s1 + 1;
        h = subplot(CH,CH,s1); 
        set(h,'FontSize',20,'FontWeight','bold');
        
        GPDC_tmp = squeeze(GPDC(i,j,:,:));        
        if(i==j)
            GPDC_tmp = zeros(1,size(GPDC,3));
        end
        area(linspace(0,Fmax,Nf),GPDC_tmp,'FaceColor',[0 0 0])
        axis([0 x_max 0 y_max])
        if(i<CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',zeros(1,0));
        elseif(i<CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',zeros(1,0));
        elseif(i==CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',[0 x_max]);
        elseif(i==CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',[0 x_max]);
        end
        if(i==CH && j==ceil(CH/2))
            xlabel('Frequency (Hz)','Fontsize',20,'FontWeight','bold')
        end
        
    end
    suptitle('gPDC') 
end

%%%%
figure, % ---> OPDC
s1 = 0;
clear mask
OPDC = abs(OPDC);

x_max = Fmax;
y_max = .3;

for i = 1 : CH
    for j = 1 : CH
        s1 = s1 + 1;
        h = subplot(CH,CH,s1); 
        set(h,'FontSize',20,'FontWeight','bold');
        
        OPDC_tmp = squeeze(OPDC(i,j,:,:));        
        if(i==j)
            OPDC_tmp = zeros(1,size(OPDC,3));
        end
        area(linspace(0,Fmax,Nf),OPDC_tmp,'FaceColor',[0 0 0])
        axis([0 x_max 0 y_max])
        if(i<CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',zeros(1,0));
        elseif(i<CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',zeros(1,0));
        elseif(i==CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',[0 x_max]);
        elseif(i==CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',[0 x_max]);
        end
        if(i==CH && j==ceil(CH/2))
            xlabel('Frequency (Hz)','Fontsize',20,'FontWeight','bold')
        end
        
    end
   suptitle('OPDC') 
end

%%%%
figure, % ---> gOPDC
s1 = 0;
clear mask
GOPDC = abs(GOPDC);

x_max = Fmax;
y_max = .3;

for i = 1 : CH
    for j = 1 : CH
        s1 = s1 + 1;
        h = subplot(CH,CH,s1); 
        set(h,'FontSize',20,'FontWeight','bold');
        
        GOPDC_tmp = squeeze(GOPDC(i,j,:,:));        
%         if(i==j)
%             GOPDC_tmp = zeros(1,size(GOPDC,3));
%         end
        area(linspace(0,Fmax,Nf),GOPDC_tmp,'FaceColor',[0 0 0])
        axis([0 x_max 0 y_max])
        if(i<CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',zeros(1,0));
        elseif(i<CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',zeros(1,0));
        elseif(i==CH && j>1)
            set(h,'YTick',zeros(1,0),'XTick',[0 x_max]);
        elseif(i==CH && j==1)
            set(h,'YTick',[0 y_max],'XTick',[0 x_max]);
        end
        if(i==CH && j==ceil(CH/2))
            xlabel('Frequency (Hz)','Fontsize',20,'FontWeight','bold')
        end
        
    end
    suptitle('GOPDC')
end




























