% Yuan Gao, Rice University

% run('/Users/gaoyuan/Documents/MATLAB/vlfeat-0.9.20/toolbox/vl_setup')
close all
clear
FEAT = struct();
test_result = zeros(3,3);
BUDDHA = dir('/Users/gaoyuan/Documents/MATLAB/buddha');
BUTTERFLY = dir('/Users/gaoyuan/Documents/MATLAB/butterfly');
AIRPLANES = dir('/Users/gaoyuan/Documents/MATLAB/airplanes');
[numOfBuddha,~] = size(BUDDHA);
[numOfFly,~] = size(BUTTERFLY);
[numOfPlane,~] = size(AIRPLANES);
numOfCluster = 1000;
bin = (1:numOfCluster);
threshPoint = 0.9;
root = '/Users/gaoyuan/Documents/MATLAB/';

% initialization for later concatenation
d_buddha = []; d_butterfly = []; d_airplane = [];

% first three entries are MAC temp files
for i = 1:1:numOfBuddha
    if (BUDDHA(i).name(1) == '.')
        continue;
    end
    % extract features for each of the classes  
    I_buddha = im2double(imread(strcat(root,...
        'buddha/',BUDDHA(i).name)));
    if (ndims(I_buddha) == 3)
        I_buddha = rgb2gray(I_buddha);
    end
    pts = detectSURFFeatures(I_buddha,'NumOctaves',2);
    [d_buddha0,~] = extractFeatures(I_buddha,pts);
    d_buddha = [d_buddha d_buddha0'];
end

for i = 1:1:numOfFly
    if (BUTTERFLY(i).name(1) == '.')
        continue;
    end
    I_butterfly = im2double(imread(strcat(root,...
        'butterfly/', BUTTERFLY(i).name)));
    if (ndims(I_butterfly) == 3)
        I_butterfly = rgb2gray(I_butterfly);
    end
    pts = detectSURFFeatures(I_butterfly,'NumOctaves',2);
    [d_butt0,~] = extractFeatures(I_butterfly,pts);
    d_butterfly = [d_butterfly d_butt0'];
end

for i = 1:1:numOfPlane
    if (AIRPLANES(i).name(1) == '.')
        continue;
    end
    I_airplanes = im2double(imread(strcat(root,...
        'airplanes/',AIRPLANES(i).name)));
    if (ndims(I_airplanes) == 3)        
        I_airplanes = rgb2gray(I_airplanes);
    end
    pts = detectSURFFeatures(I_airplanes,'NumOctaves',2);
    [d_air0,~] = extractFeatures(I_airplanes,pts);
    d_airplane = [d_airplane d_air0'];
end

% the total is the complete pool of descriptors
featureBag = double([d_buddha d_butterfly d_airplane]);
% transform it into a row for concatenation on top
[centers, ~] = vl_kmeans(featureBag, numOfCluster,'distance','l2');
kdtree = vl_kdtreebuild(centers,'thresholdmethod','median');

% set points out of threshold to 0 to exclude from bin 1~1000
[buddha_match_idx, buddha_D] = vl_kdtreequery(kdtree,...
    centers, double(d_buddha));
buddha_thresh = max(buddha_D) * threshPoint;
buddha_match_idx(buddha_D > buddha_thresh) = 0;

[fly_match_idx, fly_D] = vl_kdtreequery(kdtree,...
    centers, double(d_butterfly));
fly_thresh = max(fly_D) * threshPoint;
fly_match_idx(fly_D > fly_thresh) = 0;

[plane_match_idx, plane_D] = vl_kdtreequery(kdtree,...
    centers, double(d_airplane));
plane_thresh = max(plane_D) * threshPoint;
plane_match_idx(plane_D > plane_thresh) = 0;

buddha_count = histc(buddha_match_idx, bin);
butterfly_count = histc(fly_match_idx, bin);
airplane_count = histc(plane_match_idx, bin);

% figure
% bar(buddha_count);
% xlim([1 numOfCluster]);
% title('Occurrence of Visual Words in Buddha');
% figure
% bar(butterfly_count);
% xlim([1 numOfCluster]);
% title('Occurrence of Visual Words in Butterfly');
% figure
% bar(airplane_count);
% xlim([1 numOfCluster]);
% title('Occurrence of Visual Words in Airplane');

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
xlim([1 numOfCluster]);
title('Normalized Visual Words in Buddha');
figure
bar(norm_butterfly_count);
xlim([1 numOfCluster]);
title('Normalized Visual Words in Butterfly');
figure
bar(norm_airplane_count);
xlim([1 numOfCluster]);
title('Normalized Visual Words in Airplane');

% Now enter the testing phase
BUDDHA_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_1');
BUTTERFLY_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_2');
AIRPLANES_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_3');
TRAINED_DATA = [FEAT.BUDDHA; FEAT.BUTTERFLY; FEAT.AIRPLANES];

[numOfBuddha, ~] = size(BUDDHA_TEST);
[numOfButterfly, ~] = size(BUTTERFLY_TEST);
[numOfAirplane, ~] = size(AIRPLANES_TEST);

% Store descriptors for each single image of each class
for i = 1:1:numOfBuddha
    if (BUDDHA_TEST(i).name(1) == '.')
        continue;
    end
    I_buddha_test = im2double(imread(strcat(root,...
        'TestDataset_1/',BUDDHA_TEST(i).name)));
    if (ndims(I_buddha_test) == 3)
        I_buddha_test = rgb2gray(I_buddha_test);
    end
    pts = detectSURFFeatures(I_buddha_test,'NumOctaves',2);
    [d_buddha0_test,~] = extractFeatures(I_buddha_test,pts);
    [buddha_match_test_idx, buddha_D_test] = ...
        vl_kdtreequery(kdtree, centers, double(d_buddha0_test'));
    buddha_thresh_test = max(buddha_D_test) * threshPoint;
    buddha_match_test_idx(buddha_D_test > buddha_thresh_test) = 0;
    
    buddha_test_count = histc(buddha_match_test_idx, bin);
    norm_buddha_test_count = buddha_test_count/norm(buddha_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, norm_buddha_test_count,...
        'Distance','euclidean');
    if (class == 1)
        test_result(1,1) = test_result(1,1) + 1;
    elseif (class == 2)
        test_result(1,2) = test_result(1,2) + 1;
    else
        test_result(1,3) = test_result(1,3) + 1;
    end
end

for i = 1:1:numOfButterfly
    if (BUTTERFLY_TEST(i).name(1) == '.')
        continue;
    end
    I_butterfly_test = im2double(imread(strcat(root,...
        'TestDataset_2/', BUTTERFLY_TEST(i).name)));
    if (ndims(I_butterfly_test) == 3)
        I_butterfly_test = rgb2gray(I_butterfly_test);
    end
    pts = detectSURFFeatures(I_butterfly_test,'NumOctaves',2);
    [d_butt0_test,~] = extractFeatures(I_butterfly_test,pts);
    [fly_match_test_idx, fly_D_test] = ...
        vl_kdtreequery(kdtree, centers, double(d_butt0_test'));
    fly_thresh_test = max(fly_D_test) * threshPoint;
    fly_match_test_idx(fly_D_test > fly_thresh_test) = 0;
    
    fly_test_count = histc(fly_match_test_idx, bin);
    norm_fly_test_count = fly_test_count/norm(fly_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, norm_fly_test_count,...
        'Distance','euclidean');
    if (class == 1)
        test_result(2,1) = test_result(2,1) + 1;
    elseif (class == 2)
        test_result(2,2) = test_result(2,2) + 1;
    else
        test_result(2,3) = test_result(2,3) + 1;
    end 
end

for i = 1:1:numOfAirplane
    if (AIRPLANES_TEST(i).name(1) == '.')
        continue;
    end
    I_airplanes_test = im2double(imread(strcat(root,...
        'TestDataset_3/',AIRPLANES_TEST(i).name)));
    if (ndims(I_airplanes_test) == 3)        
        I_airplanes_test = rgb2gray(I_airplanes_test);
    end
    pts = detectSURFFeatures(I_airplanes_test,'NumOctaves',2);
    [d_air0_test,~] = extractFeatures(I_airplanes_test,pts);
    [air_match_test_idx, air_D_test] = ...
        vl_kdtreequery(kdtree, centers, double(d_air0_test'));
    air_thresh_test = max(air_D_test) * threshPoint;
    air_match_test_idx(air_D_test > air_thresh_test) = 0;
    
    air_test_count = histc(air_match_test_idx, bin);
    norm_air_test_count = air_test_count/norm(air_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, norm_air_test_count,...
        'Distance','euclidean');
    if (class == 1)
        test_result(3,1) = test_result(3,1) + 1;
    elseif (class == 2)
        test_result(3,2) = test_result(3,2) + 1;
    else
        test_result(3,3) = test_result(3,3) + 1;
    end   
end

test_result(1,:) = test_result(1,:)/sum(test_result(1,:));
test_result(2,:) = test_result(2,:)/sum(test_result(2,:));
test_result(3,:) = test_result(3,:)/sum(test_result(3,:));
test_result
