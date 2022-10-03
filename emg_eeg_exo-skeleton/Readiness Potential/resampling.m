fs =500;
% left leg
left_leg = readtable('210414141533_left_edited.txt');
left_leg =left_leg(584:end,:);
AA_C = table2cell(left_leg); 
x_left = AA_C(:,8);
x_left = str2double(strrep(x_left,',','.'));
tx_left =AA_C(1:51601,2); 


% right leg
right_leg = readtable('210414141544_right_edited.txt');
right_leg =right_leg(20:end,:);
AA_C1 = table2cell(right_leg);
x_right = AA_C1(:,8); 
x_right = str2double(strrep(x_right,',','.'));
tx_right =AA_C1(:,2); 

tx =[tx_left tx_right];
% y = resample(x,tx,fs)
% t_final = datetime(tx,'Format','HH:mm:ss.SSS'); 
ms_left =[]
ms_right=[]
for j = [1 2]
    ms = [];
    for i =1:length(tx_right)
        Hour_ms = str2num(tx{i,j}(1:2))*3.6e6;
        min_ms = str2num(tx{i,j}(4:5))*6e4;
        sec_ms = str2num(tx{i,j}(7:8))*1000;
        MS = str2num(tx{i,j}(10:12));
        ms_total = Hour_ms + min_ms + sec_ms + MS;
        ms = [ms ;ms_total];
       
    end
    if j == 1
        ms_left = [ms_left ;ms];
    end
    if j==2
        ms_right = [ms_right ;ms];
    end
        
end

y = resample(x_left(1:51601), ms_left./1000, 100);
y1 = resample(x_right,ms_right./1000,100);

% z = resample(x_left, ms_left./1000, 500);
% z1 = resample(x_right(1:46381),ms_right./1000,500);

writematrix(y,'210414141533_left_leg_edited_100Hz.txt');
writematrix(y1,'210414141544_right_leg_edited_100Hz.txt');

figure (1)
plot(y)
hold on
plot(y1)

figure (2)
plot(x_left)
hold on
plot(x_right)

