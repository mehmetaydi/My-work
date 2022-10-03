clear all
clc
% close all
warning off 

%% Load data

% load ContrAED4_VEP_ArtifactRejected_Left % A sample EEG dataset with the size of N_epochs x L_epoch x CH
                                          % where N_epochs is the number of epochs, L_epoch is the length of each
                                          % epoch in samples and CH is the number of channels.

% [bcc edf_file] = edfread('20210409102517_Patient01.edf'); 

% The sample interval where movement happens these values were taken from  python for right leg movement
file = input('Enter data frame that you want to execute \n press 1 if you want to visualize files102517 ( Motion plus)  \n press 2 if you want to visualize files104849 (Therapy plus ) \n press 3 if you want to visualize data recorded at 135143   \n press 4 if you want to visualize recorded at 141523    \n press 5 if you want to visualize recorded at 143751 :')
if file==1
    file_name = 'Motion plus';
    easy_file = importdata('20210409102517_Patient01.easy'); % easy_file_102517
    segments ={2796:2995};

%     segments ={2697:4830 7982:10152 19048:21809 24022:26528 33669:36837 39727:42587 48175:53269 57962:62273 78718:82006};
end
if file==2
    file_name = 'therapy plus';
    easy_file = importdata('20210409104849_Patient01.easy'); % easy_file_102517
    segments ={1377:3897 5821:10493 17416:20833 22756:26444 30976:34101 36073:39354 59007:62187 65938:69991 81794:85676 94355:98989};  
end


if file==3
    file_name = '135143';
    easy_file = importdata('20210414135143_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(8259:end,:);
    segments ={9832:14020 15914:19340 26884:31110 32693:36962 50217:54405 56052:59551 64599:69894 73151:78423 82931:85422};
end
if file==4
    file_name = '141523';
    easy_file = importdata('20210414141523_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(10597:end,:);
    segments ={6380:10595 12525:16090 20790:29302 32373:40572 50839:54439 56262:59753 65222:68742 73065:76879 86487:90712 93620:96744};  
end

if file==5
    file_name = '143751';
    easy_file = importdata('20210414143751_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(9777:end,:);
    segments ={507:3014 3520:5280 7650:9193 9912:11482 12071:12994};
end


                                          
[xd, K, time] = epoch_creator_easy_file(segments,easy_file,2, 0,100); 
vv = epoch(easy_file, 100,0.5,0.5); 
segment_number =length(K);
s2 =0;
%%
for tt = 1:1
    t =K{:,:,1,tt};
    s2=s2+1;
    T =cell2mat(t);
    erp_al = T  ; 
    erp_all = vv  ; 
    

    %% EEG data
    % erp_all = erp_all ./ max(abs(erp_all)) ;  
    [N_epochs L_epoch CH] = size(erp_all);
%     N_epochs = 3; % In order to reduce the processing time, only 20 epochs are used in this sample script. You can use the whole dataset by commenting this line.

    %% Connectivity analysis
    Fs = 100; % Sampling frequency
    %erp = squeeze(mean(erp_all,1));
    erp = squeeze(mean(erp_all,1));
 %%   
    m = mean(erp,1)'; % Ensemble mean of all epochs
    erp_ave = erp - repmat(m,1,L_epoch)'; % Averaged ERP

    t1 = fix(0*Fs);   % Start time of the original time segment (between 0 and 1000msec)
    t2 = fix(0.5*Fs); % End time of the original time segment (between 0 and 1000msec)
    t3 = fix(0.5*Fs); % Start time of the thresholding time segment (between 0 and 1000msec)
    t4 = fix(1*Fs);   % End time of the thresholding time segment (between 0 and 1000msec)
    Fmax = 40;        % Maximum frequency band in the T-F GOPDC measure
    Nf = 2*Fmax;      % Number of frequency bins from 0 to Fmax

    PDC_0To400msec = zeros(CH,CH,Nf,t2,N_epochs);
    PDC_600To1000msec_surr = zeros(CH,CH,Nf,t2,N_epochs);

    for epochs = 1 : N_epochs

        y = squeeze(erp_all(epochs,:,:)); % each epoch

        %% DEKF for time-varying MVAR model estimation
        inp_model.data = y;
        inp_model.order = 5;             % Model order
        [A,C] = DEKF3(inp_model);        % Estimated time-varying parameters, A = [A1 A2 ... Ar] --> A: CH x CH*p x L

        %% Compute connectivity measures including GOPDC
        [GPDC,OPDC,PDC,GOPDC,S] = PDC_dDTF_imag(A,C,inp_model.order,Fs,Fmax,Nf);
    %     disp(['The original signal --> Finished - epoch ' num2str(epochs)])

        %% Divide the GOPDC plots into two parts original
        PDC_0To400msec(:,:,:,:,epochs) = PDC(:,:,:,t1+1:t2); % GOPDC of the main part of the VEP (first 400msec interval)
        PDC_600To1000msec_surr(:,:,:,:,epochs) = PDC(:,:,:,t3+1:t4); % GOPDC of the last part of the VEP (last 400msec interval)-->for thresholding

    %     %% Divide the GOPDC plots into two parts easy file
    %     PDC_0To400msec(:,:,:,:,epochs) = GOPDC(:,:,:,t1+1:t2); % GOPDC of the main part of the VEP (first 400msec interval)
    %     PDC_600To1000msec_surr(:,:,:,:,epochs) = GOPDC(:,:,:,t3+1:t4-1); % GOPDC of the last part of the VEP (last 400msec interval)-->for thresholding
    end

    %% Estimate the thresholding plane
    hist_resolution = 50;
    orig_hist = zeros(N_epochs,hist_resolution); % Values of each epoch
    orig_xout = zeros(N_epochs,hist_resolution);
    surr_thresh = zeros(CH,CH,Nf,t2);
    s = 1;
    pdf_surr_all = zeros(CH*CH*Nf*t2,hist_resolution);
    for i = 1 : CH
        for j = 1 : CH
            if(i~=j)
                for f = 1: Nf
                    for t = 1 : t2
                        %%%%%%%%%%%%
                        orig_dist = abs(PDC_600To1000msec_surr(i,j,f,t,:));
                        orig_dist = abs(orig_dist(:));
                        %%%%%%%%%%%% Estimation of the null distribution (%99 pecentile)
                        surr_thresh(i,j,f,t) = prctile(orig_dist,99); % Threshold of the surrpgates at 99% confidence interval
                        %%%%%%%%%%%%%
                        pdf_surr_all(s,:) = hist(orig_dist,hist_resolution);
                        s = s + 1;
                    end
                end
            end
        end
       
    end

    %% Thresholding and averaging over epochs
    PDC_ave = zeros(CH,CH,Nf,t2);
    for i = 1 : N_epochs
        clear mask
        PDC_tmp = squeeze(PDC_0To400msec(:,:,:,:,i));
        mask = PDC_tmp>surr_thresh;
        PDC_ave = PDC_ave + PDC_tmp.*mask;
    end
    PDC_ave = PDC_ave/N_epochs;

    %% Plot GOPDC figures
    %%% labels
    ch_label{1}='EMG1';
    % ch_label{2}='EMG2';
%     ch_label{2}='C1';
    ch_label{2}='C2';
    ch_label{3}='C3';
    ch_label{4}='C4';
    ch_label{5}='Cz';
    % ch_label{8}='P4';
    % ch_label{9}='O1';
    % ch_label{10}='O2';



    % img_all =[];
    s1 =0;
    for i = 1 : CH

        for j = 1 : CH
            s1 =s1+1;
            PDC_tmp = abs(PDC_ave(i,j,:,:));
            img = squeeze(PDC_tmp);
%             if(i==j)
%                 img = zeros(size(PDC_ave,3),size(PDC_ave,4));
%             end
    %         img_all =[img_all;img];
            k(:,:,s1) =img;
        end 
        
    end
   all(:,:,:,s2) =k; 
   disp([ num2str(tt) '. segment done'])
end
    %%

% Information flow from  channels to EMG
    s =0;
    figure, % ---> GOPDC
    clear mask
    for i = 1 : segment_number
        for j = 1 : CH
            s = s + 1;
            h = subplot(segment_number,CH,s); 
            set(h,'FontSize',15,'FontWeight','bold');
            imagesc([100 1000*time],[2 Fmax],all(:,:,j,i))         
            colormap(jet(256))
            caxis([0 .05])
            set(h,'YDir','normal')
            grid on
            if(i==segment_number && j==ceil(CH/2))
                xlabel('Time (msec)','Fontsize',20)
            end
            if(i==ceil(segment_number/2) && j==1)
                ylabel('Frequency (Hz)','Fontsize',20)
            end
            title([ch_label{1} ' <--- ' ch_label{j}  ],'Fontsize',15,'FontWeight','bold')
            
%             saveas(h,sprintf('FIG1.pdf'));

        end
        suptitle (file_name)
        
    end
    
    h2 = colorbar;
    set(h2, 'Position', [.92 .11 .03 .8150],'FontSize',20,'FontWeight','bold')
  
%%
% Information flow from EMG to channels 
    s =0;
    figure, % ---> GOPDC
    clear mask
    for i = 1 : segment_number
        k =1;
        for j = 1 : CH
            s = s + 1;
            
            h = subplot(segment_number,CH,s);
            set(h,'FontSize',15,'FontWeight','bold');
            imagesc([0 1000*time],[2 Fmax],all(:,:,k,i))         
            colormap(jet(256))
            caxis([0 .05])
            
            k = k+5;
            set(h,'YDir','normal')
            grid on
            if(i==segment_number && j==ceil(CH/2))
                xlabel('Time (msec)','Fontsize',20)
            end
            if(i==ceil(segment_number/2) && j==1)
                ylabel('Frequency (Hz)','Fontsize',20)
            end
            title([ch_label{j} ' <--- ' ch_label{1}  ],'Fontsize',15,'FontWeight','bold')
            

        end
        suptitle (file_name)
    end
    
    h2 = colorbar;
    set(h2, 'Position', [.92 .11 .03 .8150],'FontSize',20,'FontWeight','bold')
    %%
    
    % All together

% clear mask
% for t =1:segment_number
%     figure(t)
%     s1 = 0;
%     for i = 1 : CH
%         m =1;
%         for j = 1 : CH
%             s1 = s1 + 1;
%             h = subplot(CH,CH,s1);
%             set(h,'FontSize',15,'FontWeight','bold');
%             PDC_tmp = abs(PDC_ave(i,j,:,:));
%             img = squeeze(PDC_tmp);
%     %         if(i==j)
%     %             img = zeros(size(PDC_ave,3),size(PDC_ave,4));
%     %         end
% 
%             imagesc([0 1000*time],[2 Fmax],all(:,:,m,t))
%             colormap(jet(256))
%             caxis([0 .05])
%             m =m+1;
%             set(h,'YDir','normal')
%             grid on
%             if(i==CH && j==ceil(CH/2))
%                 xlabel('Time (msec)','Fontsize',20)
%             end
%             if(i==ceil(CH/2) && j==1)
%                 ylabel('Frequency (Hz)','Fontsize',20)
%             end
%             title([ch_label{i} ' <--- ' ch_label{j}],'Fontsize',15,'FontWeight','bold')
%         end
%     end
% end
h2 = colorbar;
set(h2, 'Position', [.92 .11 .03 .8150],'FontSize',20,'FontWeight','bold')

