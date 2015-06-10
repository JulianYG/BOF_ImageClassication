% Yuan Gao, Rice University

test_result = zeros(3,3);

BUDDHA_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_1');
BUTTERFLY_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_2');
AIRPLANES_TEST = dir('/Users/gaoyuan/Documents/MATLAB/TestDataset_3');

TRAINED_DATA = [FEAT.BUDDHA; FEAT.BUTTERFLY; FEAT.AIRPLANES];

[numOfBuddha w] = size(BUDDHA_TEST);
[numOfButterfly w] = size(BUTTERFLY_TEST);
[numOfAirplane w] = size(AIRPLANES_TEST);
% Initialization
[f_buddha_test,d_buddha_test] = vl_sift(single...
    (rgb2gray(imread(strcat(root,...
        'TestDataset_1/',BUDDHA_TEST(4).name)))));
[f_butterfly_test,d_butterfly_test] = vl_sift(single(...
    rgb2gray(imread(strcat(root,...
        'TestDataset_2/', BUTTERFLY_TEST(4).name)))));
[f_airplane_test,d_airplane_test] = vl_sift(single(rgb2gray(...
    imread(strcat(root,'TestDataset_3/',AIRPLANES_TEST(4).name)))));

% first three entries are MAC temp files
% Store descriptors for each single image of each class
for i = 4:1:numOfBuddha
    I_buddha_test = im2double(imread(strcat(root,...
        'TestDataset_1/',BUDDHA_TEST(i).name)));
    if (ndims(I_buddha_test) == 3)
        [f_buddha0_test,d_buddha0_test] = ...
            vl_sift(single(rgb2gray(I_buddha_test)));
    else
        [f_buddha0_test,d_buddha_test] = vl_sift(single(I_buddha_test));
    end
    
    [buddha_match_test_idx buddha_D_test] = ...
        knnsearch(centers', double(d_buddha0_test)');
    buddha_thresh_test = max(buddha_D_test) * threshPoint;
    buddha_match_test_idx(buddha_D_test > buddha_thresh_test) = 0;
    
    buddha_test_count = histc(buddha_match_test_idx', bin);
    norm_buddha_test_count = buddha_test_count/norm(buddha_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, ...
        norm_buddha_test_count,'Distance','correlation');
    if (class == 1)
        test_result(1,1) = test_result(1,1) + 1;
    elseif (class == 2)
        test_result(1,2) = test_result(1,2) + 1;
    else
        test_result(1,3) = test_result(1,3) + 1;
    end
end

for i = 4:1:numOfButterfly
    I_butterfly_test = im2double(imread(strcat(root,...
        'TestDataset_2/', BUTTERFLY_TEST(i).name)));
    if (ndims(I_butterfly_test) == 3)
        [f_butt0_test,d_butt0_test] = ...
            vl_sift(single(rgb2gray(I_butterfly_test)));
    else
        [f_butt0_test,d_butt0_test] = vl_sift(single(I_butterfly_test));
    end 
    
    [fly_match_test_idx fly_D_test] = ...
        knnsearch(centers', double(d_butt0_test)');
    fly_thresh_test = max(fly_D_test) * threshPoint;
    fly_match_test_idx(fly_D_test > fly_thresh_test) = 0;
    
    fly_test_count = histc(fly_match_test_idx', bin);
    norm_fly_test_count = fly_test_count/norm(fly_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, ...
        norm_fly_test_count,'Distance','correlation');
    if (class == 1)
        test_result(2,1) = test_result(2,1) + 1;
    elseif (class == 2)
        test_result(2,2) = test_result(2,2) + 1;
    else
        test_result(2,3) = test_result(2,3) + 1;
    end 
end

for i = 4:1:numOfAirplane
    I_airplanes_test = im2double(imread(strcat(root,...
        'TestDataset_3/',AIRPLANES_TEST(i).name)));
    if (ndims(I_airplanes_test) == 3)        
        [f_air0_test,d_air0_test] = ...
            vl_sift(single(rgb2gray(I_airplanes_test)));
    else
        [f_air0_test,d_air0_test] = vl_sift(single(I_airplanes_test));
    end
    
    [air_match_test_idx air_D_test] = ...
        knnsearch(centers', double(d_air0_test)');
    air_thresh_test = max(air_D_test) * threshPoint;
    air_match_test_idx(air_D_test > air_thresh_test) = 0;
    
    air_test_count = histc(air_match_test_idx', bin);
    norm_air_test_count = air_test_count/norm(air_test_count, 1);
    % get the histogram vectors and choose closest one
    class = knnsearch(TRAINED_DATA, ...
        norm_air_test_count,'Distance','correlation');
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