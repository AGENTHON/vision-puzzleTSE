clear all;
close all;
clc;

image = imread('test.png');

figure;
imshow(image);
% subplot(2,2,1);imshow(image);
% subplot(2,2,2);imshow(image(:,:,1));
% subplot(2,2,3);imshow(image(:,:,2));
% subplot(2,2,4);imshow(image(:,:,3));
%%

IG = rgb2gray(image);
figure;
imshow(IG)
idx=find(IG==240);
IG(idx)=0;

figure;
imshow(IG);

IL = IG>0;
IL = imfill(IL,'holes');

figure;
imshow(IL);

L = bwlabel(IL);

figure;
imshow(L);


%% Test rgb2hsv

imH = rgb2hsv(image);
figure;
subplot(2,2,1);imshow(imH);
subplot(2,2,2);imshow(imH(:,:,1));
subplot(2,2,3);imshow(imH(:,:,2));
subplot(2,2,4);imshow(imH(:,:,3));

imH1 = imH(:,:,1);
imH2 = imH(:,:,2);
imH3 = imH(:,:,3);

% imTest = imH2.*imH3;
imTest= imH2;
figure;
imshow(imTest);

thresh = graythresh(imTest);
imS = imTest>thresh;

figure;
imshow(imS);

%% Start 
% X =double(rgb2gray(image));
X =double(imH2);
Bx = [-1,0,1;-2,0,2;-1,0,1]; % Sobel Gx kernel
By = Bx'; % gradient Gy
Yx = filter2(Bx,X); % convolve in 2d
Yy = filter2(By,X);
G = sqrt(Yy.^2 + Yx.^2); % Find magnitude
Gmin = min(min(G));
dx = max(max(G)) - Gmin; % find range
G = floor((G-Gmin)/dx*255); % normalise from 0 to 255
figure;
imshow(G,[]);

G2 = G>thresh*20;



G2 = bwareaopen(G2,100);
G2 = imfill(G2,'holes');


figure;
imshow(G2,[]);
%%
% I2 = rgb2gray(image);
% idx=find(imS==1);
% I2(idx)=255;
% 
% figure;
% imshow(I2)
% 
% 
% thresh = graythresh(I2);
% imS2 = I2>255*thresh;
% figure;
% imshow(imS2);

imS3 = imS + G2;
figure;
imshow(imS3);

% %%
imS3 = imclose(imS3,strel('disk',5));
imS3 = imfill(imS3,'holes');
imS3 = bwareaopen(imS3,100);
figure;
imshow(imS3);title('final');


