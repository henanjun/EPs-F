function [mean_matrix,sup_img,indexes] = mean_feature (img,labels)
%=================================================================================
%This function is used to extract spatial information (mean feature) within each superpixel 
%input arguments:  img          : dimension-reduced HSI
%                  labels       : superpixel segmentation map
%output arguments: maen_matrix  : feature matrix within each superpixel 
%                  sup_img      : superpixels image
%                  indexes      : pixels indexes of each sperpixels
%=================================================================================
[ no_rows,no_lines, no_bands] = size(img);
img=reshape(img,[no_rows*no_lines,no_bands]);
%img = ToVector(img);
img = img';
mean_matrix=img;
MaxSegments=max(labels(:));
indexes={};
sup_img=[];
for i=0:MaxSegments
    supind=find(labels==i);
    v=img(:,supind);
    meanv=mean(v,2);
    indexes{i+1}=supind;
    sup_img(:,i+1) = meanv;
    [a,~]=size(supind);
    for j=1:a
        mean_matrix(:,supind(j))=meanv;
    end
end
mean_matrix=mean_matrix';
mean_matrix=reshape(mean_matrix,no_rows,no_lines, no_bands);
    