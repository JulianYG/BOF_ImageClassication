% Yuan Gao, Rice University

%% Training phase
close all
clear
root = '/Users/gaoyuan/Documents/MATLAB/';
numOfCluster = 6750;
featureBag = [];
confusionMatrix = zeros(25,25);
CLASS_IDX = cell(25,1);
bin = (1:numOfCluster);
TRAINING_SET = struct();
TRAINING_DATA = dir(strcat(root,'TrainingDataset/'));
[numOfClass, ~] = size(TRAINING_DATA);
% load all images from testing data 
% this creates a nested struct of struct
for f = 4: 1: numOfClass
    TRAINING_SET(f - 3).CLASS = dir(strcat(root,...
        'TrainingDataset/', TRAINING_DATA(f).name));
    CLASS_IDX{f - 3} = strtok(TRAINING_DATA(f).name, '.');
end

for c = 1: 1: numOfClass - 3
    [numOfImage, ~] = size(TRAINING_SET(c).CLASS);
    d_class_pool = [];
    for i = 1: 1: numOfImage
        if (TRAINING_SET(c).CLASS(i).name(1) == '.')
            continue;
        end
        image = im2double(imread(...
            strcat(root,'TrainingDataset/',TRAINING_DATA(c + 3).name,...
                '/', TRAINING_SET(c).CLASS(i).name)));
        if (ndims(image) == 3)   
            image = rgb2gray(image);
        end
        pts = detectSURFFeatures(image,'NumOctaves',6);
        [d,~] = extractFeatures(image,pts,'Method','SURF');   
        d_class_pool = [d_class_pool d'];        
    end
    TRAINING_SET(c).DESCRIPTORS = d_class_pool;
    % gathering all features from all pictures
end
% now the most time consuming part to extract clusters

[centroids, ~] = vl_kmeans(featureBag, numOfCluster,...
    'distance','l2','algorithm','ann');
forest = vl_kdtreebuild(centroids,'numtrees',3,'thresholdmethod','mean');

% Loop to generate bin for each class
feat_pool = zeros(numOfClass - 3, numOfCluster);
for c = 1: 1: numOfClass - 3
    [match_idx, dist] = vl_kdtreequery(forest,...
        centroids, TRAINING_SET(c).DESCRIPTORS);
    threshold = prctile(dist,90);
    match_idx(dist > threshold) = 0;

    class_cnt = histc(match_idx,bin);
    n_class_cnt = class_cnt/sum(class_cnt);
    % strictly following the order of classes
    feat_pool(c,:) = n_class_cnt;
end

%% Testing phase
TEST_SET = dir(strcat(root,'TestDataset/'));
[numOfImage, ~] = size(TEST_SET);
for j = 1: 1: numOfImage
    filename = TEST_SET(j).name;
    % avoid temp/directory files
    if (filename(1) == '.')
        continue;
    end
    test_image = im2double(imread(strcat(root,...
        'TestDataset/', filename)));
    if (ndims(test_image) == 3)
        test_image = rgb2gray(test_image);
    end
    pts = detectSURFFeatures(test_image,'NumOctaves',6);
    [d_test,~] = extractFeatures(test_image,pts,'Method','SURF');
 
    % generate the bin hist for each test image
    [test_match_idx, test_dist] = vl_kdtreequery(forest,...
        centroids, d_test');
    thresh_test = prctile(dist,90);
    test_match_idx(test_dist > thresh_test) = 0;
    test_cnt = histc(test_match_idx,bin);
    n_test_cnt = test_cnt/sum(test_cnt);
    % generate confusion matrix by matching actual and predictions
    prediction = knnsearch(feat_pool, n_test_cnt,...
        'Distance','euclidean');
    actual = find(strcmp(strtok(TEST_SET(j).name, '_'), CLASS_IDX));
    confusionMatrix(actual,prediction) =...
        confusionMatrix(actual,prediction) + 1;
end

for k = 1:1:25
    confusionMatrix(k,:) = confusionMatrix(k,:)/sum(confusionMatrix(k,:));
end
avgRate = sum(diag(confusionMatrix))/25
