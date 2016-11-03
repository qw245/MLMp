function [P_est,A_est,M_est,obj]=MLMp(M_est,X_obs,NoisePower)
%% Input: 
%        1)observation: X_obs 
%        2)endmember matrix: M_est
%        3)the threshold to stop the algorithm: NoisePower
%% Output:
%        the estimated P_est, A_est and M_est

%Initialize the P matrix with zeros
P_est=0.0*ones(1,size(X_obs,2)); 
%Initialize the A matrix with ADMM
A_est=sunsal(M_est,X_obs,'lambda',0,'POSITIVITY','yes','ADDONE','yes','AL_ITERS',500,'TOL',1e-200,'X_SOL',0,'CONV_THRE',0);
% A_est=M_est\X_obs;
% A_est=(SimplexProj(A_est'))';

n_b=size(M_est,1);
k_max=5000;
lag=5; % the lag for computing the objective
obj=zeros(1,k_max/10);
% Err_A=zeros(1,k_max);
% Err_M=zeros(1,k_max);
% Err_P=zeros(1,k_max);
num_unchg=zeros(1,k_max);
% P_set=zeros(1,k_max);
for k=1:k_max
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Update Abundances: ADMM scheme
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [A_est,num_unchg(k)]=abun_est(A_est,M_est,X_obs,P_est);% 
%     A_est=A_ref;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Update P: Least squares + Projection to convex set
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    P_est=prob_est(M_est*A_est,X_obs);  %P_set(k)=mean2(P_est);
%     P_est=0*P_ref;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Update E: ADMM scheme
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     M_est = M_ref;
     M_est=end_est(M_est,X_obs,A_est,P_est); 
%     [M_est,indx] = best_permut_R3(M_est,M_ref);
%     A_est=A_est(indx,:);
    %% Calculate the errors for each variable
%     Err_P(k)=norm(P_est-P_ref,'fro');
%     Err_A(k)=norm(A_est-A_ref,'fro');
%     Err_M(k)=norm(M_est-M_ref,'fro');
    
%     figure(11);
%     plot(M_ref,'-');hold on;
%     plot(M_est,'o');hold off;
%     drawnow;
    %% Stopping rule
    if mod(k,lag)==0
        obj(k/lag)=mean2((X_obs-(repmat(1-P_est,n_b,1)+repmat(P_est,n_b,1).*X_obs).*(M_est*A_est)).^2);  
%         obj(k/lag)
       if obj(k/lag)<NoisePower || (k/lag>1 &&abs(obj(k/lag)-obj(k/lag-1))/obj(k/lag)<2e-4)%8e-4
           break;
       end
%        temp=X_obs-(repmat(1-P_est,n_b,1)+repmat(P_est,n_b,1).*X_obs).*(M_est*A_est);
%        figure(111);imshow(reshape(mean(temp,1),[n_r n_c]),[]);title('Residues')
%        figure(112);histfit(temp(:),1000);title('Distribution of Residues')
%        figure(113);imshow(reshape(P_est,n_r,n_c),[]);colorbar;title('Map of P')
    end
end
% figure(1);
% subplot(4,1,1);plot(Err_P(1:k));title('Update of P Error (L2 norm)');
% subplot(4,1,2);plot(Err_A(1:k));title('Update of Abundance Error (L2 norm)');
% subplot(4,1,3);plot(Err_M(1:k));title('Update of Endmember Error (L2 norm)');
% subplot(4,1,4);plot(obj(obj>0));hold on; plot(NoisePower*ones(size(obj(obj>0))));title('Update of Objective');
