
function output = median_filter(A,wsize)
% Median filtering is done by extracting a local patch from the input image
% and calculating its median

% Apply median filter.
dim = size(A);
output = zeros(dim);

k = (wsize - 1) / 2;

for i = 1:dim(1)
   for j = 1:dim(2)
         % Calculate local region limits
         iMin = max(i-k,1);
         iMax = min(i+k,dim(1));
         jMin = max(j-k,1);
         jMax = min(j+k,dim(2));
                  
         %%--your-code-starts-here--%%
         % Use the region limits to extract a patch from the image
         patch = A(iMin:iMax, jMin:jMax);
         
         % calculate the median value from the extracted
         median_val = median(patch(:));
         
         % local region and store it to output using correct indexing.
         output(i, j) = median_val;
         %%--your-code-starts-here--%%
   end
end
