function [M_upd]=end_est(M_ini,X_obs,A_est,P_est)
[n_b,n]=size(X_obs);
L=size(A_est,1);
Temp_Y=zeros(n_b,n,L);
M_est=M_ini;
Temp_X=repmat(P_est,[n_b 1]).*X_obs+repmat(1-P_est,[n_b 1]);
for i=1:L
    Temp_Y(:,:,i)=repmat(A_est(i,:),[n_b 1]).*Temp_X;  % Temp_Y corresponds to \tilde{A}
end

temp=reshape(permute(Temp_Y,[1 3 2]),[n_b*L n]);
Y_sum=zeros(n_b*L,L);

for j=1:L
    for i=1:L
       Y_sum((i-1)*n_b+1:i*n_b,j)=squeeze(sum(temp((i-1)*n_b+1:i*n_b,:).*Temp_Y(:,:,j),2));
    end
end
%X_obs is same for all the abundances
Temp_right=squeeze(sum(Temp_Y.*repmat(X_obs,[1 1 L]),2));
%% Restore the endmember matrix row by row
for i=1:n_b
    grad_M = M_est(i,:)*Y_sum(i:n_b:end,:)-Temp_right(i,:); % gradient for ith row
    gamma  = 1/norm(Y_sum(i:n_b:end,:),'fro'); % stepsize for ith row
    M_est(i,:) = M_est(i,:)-gamma*grad_M;
end
%% Projected to the convex set: check whether it is between 0 and 1
M_upd=min(max(M_est,0),1);
end