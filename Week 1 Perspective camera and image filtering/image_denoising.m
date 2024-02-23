%% Load test images.
% Load test images and convert to double precision in the interval [0,1].
im = double(imread('einsteinpic.jpg'))/255;
im = imresize(im, [256, 256]);

%% Generate noise 
imns = imnoise(im,'salt & pepper',0.1);  % "salt and pepper" noise
imng = im+0.05*randn(size(im));  % zero-mean Gaussian noise
imng(imng<0) = 0; imng(imng>1) = 1;

%% Apply a Gaussian filter with a standard deviation of 2.5
sigmad = 2.5;
g = fspecial('gaussian', [11 11],2.5);
gflt_imns=imfilter(imns,g);
gflt_imng=imfilter(imng,g);

%% Instead of directly filtering with g, make a separable implementation
% where you use horizontal and vertical 1D convolutions.
% Store the results again to gflt_imns and gflt_imng, use conv2 with two 1D
% filters as input (see documentation for conv2)
% The result should not change.
% See Szeliski's Book chapter 3.2.1 Separable filtering

%%--your-code-starts-here--%%
kernel = [1 2 1; 2 4 2; 1 2 1];
[U, S, V] = svd(g);
sigma = diag(S);
u = U(:, 1);
v = V(:, 1);
h = sqrt(sigma(1)) * u;
v = sqrt(sigma(1)) * v';
%reconstructed_kernel = h * v;

% Apply horizontal convolution
gflt_imns_h = conv2(imns, h, 'same');

% Apply vertical convolution
gflt_imns = conv2(gflt_imns_h, v, 'same');

% Apply horizontal convolution
gflt_imng_h = conv2(imng, h, 'same');

% Apply vertical convolution
gflt_imng = conv2(gflt_imng_h, v, 'same');

%%--your-code-ends-here--%%


%% Apply median filtering, use neighborhood size 5x5
%  Open median_filter.m and implement the missing code
%  Store the results in medflt_imns and medflt_imng.
%  Use the median_filter.m function

%%--your-code-starts-here--%%
wsize = 5;
medflt_imns = median_filter(imns, wsize);  % replace me
medflt_imng = median_filter(imng, wsize);  % replace me
%%--your-code-ends-here--%%

%% Apply bilateral filter to each image with window size 11.
% See section 3.3.1 of Szeliski's book
% Set window size to 11.
% Use sigma value 2.5 for the domain kernel and 0.1 for range kernel.

% Set bilateral filter parameters.
w     = 11;       % bilateral filter half-width, filter size = 2*w+1
sigma = [2.5, 0.1];   % sigma_d=sigma(1), sigma_r=sigma(2)

% Apply bilateral filter to each image.
bflt_imns = bilateralfilter(imns,w,sigma);
bflt_imng = bilateralfilter(imng,w,sigma);

%% Display filtering results
figure(2); clf;
set(gcf,'Name','Filtering Results');

subplot(2,4,1); imagesc(imns);
axis image; colormap gray;
title('Input Image');

subplot(2,4,2); imagesc(gflt_imns);
axis image; colormap gray;
title('Result of Gaussian Filtering');

subplot(2,4,3); imagesc(medflt_imns);
axis image; colormap gray;
title('Result of Median Filtering');

subplot(2,4,4); imagesc(bflt_imns);
axis image; colormap gray;
title('Result of Bilateral Filtering');

subplot(2,4,5); imagesc(imng);
axis image; colormap gray;
title('Input Image');

subplot(2,4,6); imagesc(gflt_imng);
axis image; colormap gray;
title('Result of Gaussian Filtering');

subplot(2,4,7); imagesc(medflt_imng);
axis image; colormap gray;
title('Result of Median Filtering');

subplot(2,4,8); imagesc(bflt_imng);
axis image; colormap gray;
title('Result of Bilateral Filtering');
