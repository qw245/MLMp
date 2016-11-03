function [abun,k]=abun_est(abun,endm,x_obs,Prob)
[~,num_end]=size(endm);
num_obs=size(x_obs,2);
% Projected gradient descent
parfor i=1:num_obs
    temp=repmat(1-Prob(i)+Prob(i)*x_obs(:,i),[1 num_end]).*endm;
    gamma=1/norm(temp'*temp,'fro');
    abun(:,i)=abun(:,i)-gamma*(temp'*(temp*abun(:,i)-x_obs(:,i)));
end
abun=(SimplexProj(abun'))';
k=0;
end