a=VideoReader('MVI_0259.MP4');
get(a)
numFrames = ceil(a.Duration);

for img = 1:1:numFrames;
    filename= strcat('frame', num2str(img), '.png');
    b = read(a, img);
    imwrite(b, filename);
end
