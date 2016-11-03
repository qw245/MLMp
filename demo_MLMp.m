%% Demo: Unsupervised Nonlinear unmixing based on multilinear mixing (MLM) model
%  Author: Qi WEI
%  Signal Processing and Communications Laboratory (SigProC),
%  Department of Engineering, University of Cambridge
%  Email: qi.wei@eng.cam.ac.uk

% Reference:
% Q. Wei, M. Chen, J-Y. Tourneret and S. Godsill, "Unsupervised nonlinear spectral 
% unmixing based on a multilinear mixing model," IEEE Trans. Geosci. and Remote Sens.,
% under review.

% If you use this code, please cite the above paper.

clear;
close all;
clc;  

load('demo_data.mat');
x= X_obs; % observed data, the size is m*n
E= M_ref; % endmember matrix, the size is m*p
% Note: m is the number of bands, n is the number of pixels, p is the number of
% endmembers
threshold=1e-4; % The threshold to stop the algorithm
% The proposed algorithm for nonlinear unmixing based on MLM model
tic;
[P_MLMp,a_MLMp,M_est,obj]=MLMp(E,x,threshold);
toc;
% linear approximation
x_LM=M_est*a_MLMp;
% reconstructed data from the model
x_MLMp=(repmat(1-P_MLMp,size(x,1),1).*x_LM)./(1-repmat(P_MLMp,size(x,1),1).*x_LM);
% reconstruction error RE
disp(['RE for MLMp: ' num2str(norm(x_MLMp-x,'fro'))])