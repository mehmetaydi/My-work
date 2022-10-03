clear
clc
% close all
warning off
%%
file = input('Enter data frame that you want to execute \n press 1 if you want to visualize files102517 ( Motion plus)  \n press 2 if you want to visualize files104849 (Therapy plus ) \n press 3 if you want to visualize data recorded at 135143   \n press 4 if you want to visualize recorded at 141523    \n press 5 if you want to visualize recorded at 143751 :');
if file==1
    file_name = 'Motion plus';
    easy_file = importdata('20210409102517_Patient01.easy'); % easy_file_102517
     easy_file = easy_file(278:end,:);
     movement_points = importdata('move_102517_Motion plus.txt');
end
if file==2
    file_name = 'therapy plus';
    easy_file = importdata('20210409104849_Patient01.easy'); % easy_file_102517
     easy_file = easy_file(209:end,:);
     movement_points = importdata('move_104849_Therapy plus.txt');
end


if file==3
    file_name = '135143';
    easy_file = importdata('20210414135143_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(8259:end,:);
    movement_points = importdata('move_135143.txt');
   
end
if file==4
    file_name = '141523';
    easy_file = importdata('20210414141523_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(10597:end,:);
    movement_points = importdata('move_141523.txt');
   
end

if file==5
    file_name = '143751';
    easy_file = importdata('20210414143751_Patient01.easy'); % easy_file_102517
    easy_file = easy_file(9777:end,:);
    movement_points = importdata('move_143751.txt');
   
end
%% Time-varying MVAR model 
Fs = 100;
Fmax = Fs/2;
Nf = 40;
p=5;
easy_file =transpose(easy_file);
%%

[F7, Pz,O1,O2, EMG1] = deal(easy_file(1,:),easy_file(2,:),easy_file(3,:),easy_file(4,:),easy_file(5,:));

[EMG2, P8, P7, FC6, F8] = deal(easy_file(6,:),easy_file(7,:),easy_file(8,:),easy_file(9,:),easy_file(10,:));

[C4, AF4, C2, Fz, C3] = deal(easy_file(11,:),easy_file(12,:),easy_file(13,:),easy_file(14,:),easy_file(15,:));

[C1, AF3, FC5, Cz, ECG] = deal(easy_file(16,:),easy_file(17,:),easy_file(18,:),easy_file(19,:),easy_file(20,:));

x = input('Enter the name of channel(s) you want to execute\n such as EMG1 C1 and so on. Put semicolon in between\n example [C1;C2;C3] :');
ch_name = input('Please rewrite channels you selected for labelling and put semicolon among them \n example ["C1";"C2";"C3"]  :'); 

ch_name =cellstr(ch_name)
%%
before = 1; % the time before movement
after = 0.5; % the time after movement
[T,filtered_signal] = epoch_maker(x,movement_points, Fs,before,after); 

[N_epochs L_epoch CH] = size(T);
% N_epochs =1;
for epochs = 1 : N_epochs
    y = squeeze(T(epochs,:,:)); % each epoch
    %% DEKF for time-varying MVAR model estimation
    inp_model.data = y;
    inp_model.order = p;
    [A,C] = DEKF3(inp_model);        % Estimated time-varying parameters, A = [A1 A2 ... Ar] --> A: CH x CH*p x L
    %% Compute connectivity measures including GOPDC
    [DTF GPDC,OPDC,PDC,GOPDC,S] = PDC_dDTF_imag(A,C,p,Fs,Fmax,Nf);
    disp(['The original signal --> Finished - epoch ' num2str(epochs)])
    
    PDC_all(:,:,:,:,epochs) =PDC;

end

%%
PDC_all = abs(PDC_all);
for k =1: epochs
    sk =0;
    for i = 1 : CH       
        for j = 1 : CH
            sk =sk+1;
            
            PDC_tmp = (PDC_all(i,j,:,:,k));

            img = squeeze(PDC_tmp);
            img1(:,:,sk) =img;
        end
        
    end
    img2(:,:,:,k) =img1;
end
%%
%% Plot GOPDC figures
%%% labels

for ps= 1:length(ch_name)
    ch_label{ps}=ch_name{ps};
end


img = squeeze(mean(img2,4));
% Plot
figure, % ---> gPDC
s1 = 0;
clear mask

for i = 1 : CH
    for j = 1 : CH
        s1 = s1 + 1;
        h = subplot(CH,CH,s1);
        set(h,'FontSize',20,'FontWeight','bold');
        
       
        
        img = img1(:,:,s1);
        if(i==j)
            img = zeros(size(img));
        end
        imagesc(img)
        xline(length(y)*(before/(before+after)),'--r',"LineWidth",2)
        colormap(jet(256));
        set(h,'YDir','normal')
        caxis([0 .75])
        grid on
        
        if(i<CH || j>1)
            set(h,'YTick',zeros(1,0),'XTick',zeros(1,0));
        end
        if(i==CH && j==ceil(CH/2))
            xlabel('Time (sample)','Fontsize',20,'FontWeight','bold')
        end
        if(i==ceil(CH/2) && j==1)
            ylabel('Frequency (Hz)','Fontsize',20,'FontWeight','bold')
        end
        title([ch_label{i} ' <--- ' ch_label{j}],'Fontsize',15,'FontWeight','bold')
    end
    suptitle (file_name)
end
h2 = colorbar;
set(h2, 'Position', [.92 .11 .03 .8150],'FontSize',20)
%%
% for i =1:length(T(1,1,:))
%     figure(i)
%     for j = 1:length(T(:,1,:))
%       
%     plot(T(j,:,i))
%     hold on 
%     end
%     xline(100)
% end
