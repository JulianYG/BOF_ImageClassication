% Yuan Gao, Rice University
% ELEC 345 HW3

% run('/Users/gaoyuan/Documents/MATLAB/vlfeat-0.9.20/toolbox/vl_setup')
close all
FEAT = struct();
BUDDHA = dir('/Users/gaoyuan/Documents/MATLAB/buddha');
BUTTERFLY = dir('/Users/gaoyuan/Documents/MATLAB/butterfly');
AIRPLANES = dir('/Users/gaoyuan/Documents/MATLAB/airplanes');
numOfFile = 53; numOfCluster = 1000;
threshPoint = 0.8;
root = '/Users/gaoyuan/Documents/MATLAB/';

% initialization for later concatenation
[f_buddha,d_buddha] = vl_sift(single...
    (rgb2gray(imread(strcat(root,'buddha/',BUDDHA(4).name)))));
[f_butterfly,d_butterfly] = vl_sift(single(...
    rgb2gray(imread(strcat(root,'butterfly/', BUTTERFLY(4).name)))));
[f_airplane,d_airplane] = vl_sift(single(rgb2gray(...
    imread(strcat(root,'airplanes/',AIRPLANES(4).name)))));

% first three entries are MAC temp files
for i = 5:1:numOfFile
    % extract features for each of the classes  
    I_buddha = im2double(imread(strcat(root,'buddha/',BUDDHA(i).name)));
    I_butterfly = im2double(imread(strcat(root,...
        'butterfly/', BUTTERFLY(i).name)));
    I_airplanes = im2double(imread(strcat(root,...
        'airplanes/',AIRPLANES(i).name)));
    
    if (ndims(I_buddha) == 3)
        [f_buddha0,d_buddha0] = vl_sift(single(rgb2gray(I_buddha)));
    else
        [f_buddha0,d_buddha0] = vl_sift(single(I_buddha));
    end
    if (ndims(I_butterfly) == 3)
        [f_butt0,d_butt0] = vl_sift(single(rgb2gray(I_butterfly)));
    else
        [f_butt0,d_butt0] = vl_sift(single(I_butterfly));
    end 
    if (ndims(I_airplanes) == 3)        
        [f_air0,d_air0] = vl_sift(single(rgb2gray(I_airplanes)));
    else
        [f_air0,d_air0] = vl_sift(single(I_airplanes));
    end
    
    d_buddha = [d_buddha d_buddha0];
    d_butterfly = [d_butterfly d_butt0];
    d_airplane = [d_airplane d_air0];
end

% the total is the complete pool of descriptors
featureBag = double([d_buddha d_butterfly d_airplane]);
% transform it into a row for concatenation on top
[centers, asgn] = vl_kmeans(featureBag, numOfCluster);

% clusteredFeatureBag = [indices; featureBag];

% set points out of threshold to 0 to exclude from bin 1~1000
[buddha_match_idx buddha_D] = knnsearch(centers', double(d_buddha)');
buddha_thresh = max(buddha_D) * threshPoint;
buddha_match_idx(buddha_D > buddha_thresh) = 0;

[fly_match_idx fly_D] = knnsearch(centers', double(d_butterfly)');
fly_thresh = max(fly_D) * threshPoint;
fly_match_idx(fly_D > fly_thresh) = 0;

[plane_match_idx plane_D] = knnsearch(centers', double(d_airplane)');
plane_thresh = max(plane_D) * threshPoint;
plane_match_idx(plane_D > plane_thresh) = 0;

bin = (1:numOfCluster);

buddha_count = histc(buddha_match_idx', bin);
butterfly_count = histc(fly_match_idx', bin);
airplane_count = histc(plane_match_idx', bin);

figure
bar(buddha_count);
xlim([1 1000]);
title('Occurrence of Visual Words in Buddha');
figure
bar(butterfly_count);
xlim([1 1000]);
title('Occurrence of Visual Words in Butterfly');
figure
bar(airplane_count);
xlim([1 1000]);
title('Occurrence of Visual Words in Airplane');

% L1 norm since asked to normalize by total number of features
norm_buddha_count = buddha_count/norm(buddha_count,1);
norm_butterfly_count = butterfly_count/norm(butterfly_count,1);
norm_airplane_count = airplane_count/norm(airplane_count,1);
% Descriptors for particular training classes
FEAT.BUDDHA = norm_buddha_count;
FEAT.BUTTERFLY = norm_butterfly_count;
FEAT.AIRPLANES = norm_airplane_count;

figure
bar(norm_buddha_count);
xlim([1 1000]);
title('Normalized Visual Words in Buddha');
figure
bar(norm_butterfly_count);
xlim([1 1000]);
title('Normalized Visual Words in Butterfly');
figure
bar(norm_airplane_count);
xlim([1 1000]);
title('Normalized Visual Words in Airplane');


