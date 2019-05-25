%%%%%%%%%%%%%  Demo for Fusion of Extinction Profiles for Hyperspectral Images Classification  
%===================================================================================================================================================
% Author: Nanjun He
% Affiliation: College of Electrical and Information Engineering, Hunan University, Changsha, China
% Email:henanjun@hnu.edu.cn
%===================================================================================================================================================

%% Load data set
close all;clear all;clc
addpath('data');
addpath('functions')
load Pavia_IC1_PCA
load Pavia_IC2_PCA
load Pavia_IC3_PCA
load Pavia_IC1
load Pavia_IC2
load Pavia_IC3
load PaviaU_gt
load PaviaU
img = paviaU;
no_classes=9;                                
img_IC1 = Pavia_IC1;
img_IC2 = Pavia_IC2;
img_IC3 = Pavia_IC3;

%% Fusion Stage One
%1.superpixels image contruction 
[no_rows,no_lines, no_bands] = size(img_IC1);
IC1=mat2gray(Pavia_IC1_PCA);
IC2=mat2gray(Pavia_IC2_PCA);
IC3=mat2gray(Pavia_IC3_PCA);
IC1=im2uint8(IC1);
IC2=im2uint8(IC2);
IC3=im2uint8(IC3);
lambda_prime =0.8;sigma =10; conn8 = 1;
Nonzero_map = zeros(no_rows,no_lines);
Nonzero_index =  find(paviaU_gt ~= 0);
Nonzero_map(Nonzero_index)=1;
nC_base = 4500;

[ratio1] = Edge_ratio3(IC1);
[ratio2] = Edge_ratio3(IC2);
[ratio3] = Edge_ratio3(IC3);

nC1 = nC_base*ratio1;
nC2 = nC_base*ratio2;
nC3 = nC_base*ratio3;

tic;
[Pavia_IC1_PCA_ESR] = mex_ers(double(IC1),nC1,lambda_prime,sigma,conn8);
[Pavia_IC2_PCA_ESR] = mex_ers(double(IC2),nC2,lambda_prime,sigma,conn8);
[Pavia_IC3_PCA_ESR] = mex_ers(double(IC3),nC3,lambda_prime,sigma,conn8);
t=toc;

%2. Itra-information and Inter-information Extraction 
[mean_matix1,super_img1,indexes1] = mean_feature (img_IC1,Pavia_IC1_PCA_ESR);
[ weighted_matrix1 ] = weighted_mean_feature(Pavia_IC1_PCA_ESR,img_IC1,super_img1);
[mean_matix2,super_img2,indexes2] = mean_feature (img_IC2,Pavia_IC2_PCA_ESR);
[ weighted_matrix2 ] = weighted_mean_feature( Pavia_IC2_PCA_ESR,img_IC2,super_img2);
[mean_matix3,super_img3,indexes3] = mean_feature (img_IC3,Pavia_IC3_PCA_ESR);
[ weighted_matrix3 ] = weighted_mean_feature(Pavia_IC3_PCA_ESR,img_IC3,super_img3);
in_param.nfold = 5;

rand_sample_num = ones(1,no_classes)*50;
[train_SL,test_SL,~]= GenerateSample(paviaU_gt,rand_sample_num,9);
train_img=zeros(no_rows,no_lines);          
train_img(train_SL(1,:))=train_SL(2,:);   

% 3. Composite kernel construction and Composite kernel based classification
[class_label1, out_param1] = multiple_kernels_svm(img,mean_matix1,weighted_matrix1,train_img,in_param);
[class_label2, out_param2] = multiple_kernels_svm(img,mean_matix2,weighted_matrix2,train_img,in_param);
[class_label3, out_param3] = multiple_kernels_svm(img,mean_matix3,weighted_matrix3,train_img,in_param);

label(:,1) = reshape(class_label1,[610*340,1]);
label(:,2) = reshape(class_label2,[610*340,1]);
label(:,3) = reshape(class_label3,[610*340,1]);
Tlabel = label(test_SL(1,:),:); 

%% Fusion Stage Two
fusion_result = FusionTwo(Tlabel);

%% Final Classification Result
[OA,kappa,AA,CA]= calcError(test_SL(2,:)-1,fusion_result-1,[1:no_classes]);
result = zeros(no_rows,no_lines);
result(test_SL(1,:))=fusion_result;
result(train_SL(1,:))=train_SL(2,:);
classmap=reshape(result, no_rows,no_lines);
resultmap=classmap.*Nonzero_map;
SC_MKmap=label2color(resultmap,'uni');
figure,imshow(SC_MKmap);
