
clc;
clear all; 

main_class = [];
Coordinate = [];
velocity = [];
Distribution_of_area= [];
F=1;
T=50;


for i = F:1:T
    
namefile = ['C:\Users\mehmet\Desktop\physics\last experiment\frame',num2str(i),'.jpg'];
  
frame=importdata(namefile);
  

I = rgb2lab(frame);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.000;
channel1Max = 39.530;

% Define thresholds for channel 2 based on histogram settings
channel2Min = -20.737;
channel2Max = 4.680;

% Define thresholds for channel 3 based on histogram settings
channel3Min = -4.970;
channel3Max = 23.042;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Invert mask
BW = ~BW;

% Initialize output masked image based on input image.
maskedRGBImage = frame;
%%
% Set background pixels where BW is false to zero.
maskedRGBImage(repmat(~BW,[1 1 3])) = 0;

    BW2 = bwareaopen( imfill(BW,8,'holes'),20000);
    BW3 = bwareaopen( imfill(~BW,8,'holes'),20000);
    BW2 = imclearborder(BW2);
    BW3 = imclearborder(BW3);
    
    BW4 = logical(BW2 + BW3);
    
BW = ~BW;

%% detect the film and gas fraction
gas_area=length(find(~BW==1))/207360;

film_area=length(find(~BW==0))/207360;

Distribution_of_area= [Distribution_of_area;i gas_area film_area ];
%%


% Get properties.
a = regionprops(BW4, 'Area', 'Eccentricity', 'Centroid');

number_of_bubbles = length(a);

main_class=[main_class;i number_of_bubbles];
%%
%%%%%%%% Detect the velocitiy of bubbles
Center = a.Centroid;

xcoor = Center(:,1);
ycoor = Center(:,2);

Coordinate=[Coordinate;xcoor ycoor];

%%

% figure
% subplot(2,1,1); 
% imshow(BW); hold on
%     
% subplot(2,1,2); 
% imshow(BW2); hold on

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

waitbar(i/T);   
end

for i=1:T-1
    
x=Coordinate(:,1);
y=Coordinate(:,2);

    vel= sqrt(abs(x(i+1) - x(i))^2 + abs(y(i+1) - y(i))^2) ;
    velocity=[velocity;vel];
    
end



std_Value=std(velocity);

pxs2mms = 0.0091;


for i=1:T-1
if velocity <= std_Value
     vel= sqrt(abs(x(i+1) - x(i))^2 + abs(y(i+1) - y(i))^2) ;
    velocity=[velocity;vel];
    
else
    velocity==std_Value
end
end

Vel_last =pxs2mms*mean(velocity/0.2)



% figure
% plot(Distribution_of_area(:,1),Distribution_of_area(:,2),'o','MarkerEdgeColor','none','MarkerFaceColor','k');
% hold on
% plot(Distribution_of_area(:,1),Distribution_of_area(:,3),'o','MarkerEdgeColor','none','MarkerFaceColor','k');
% title({'ratio of gas fraction and film  change over time '})
% ylabel('fraction','FontSize', 14)
% xlabel('time (s)','FontSize', 14)
% legend('X','Y','b')


% figure
% plot(velocity(:,2),velocity(:,1),'MarkerEdgeColor','none','MarkerFaceColor','k');hold on
% title({'Velocity of bubbles change over time '})
% ylabel('Velocity','FontSize', 14)
% xlabel('time (s)','FontSize', 14)













