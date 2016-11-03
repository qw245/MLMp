function [Prob]=prob_est(Y_lmm,x_obs)
temp_deno=Y_lmm-Y_lmm.*x_obs;
Prob=sum(temp_deno.*(Y_lmm-x_obs),1)./sum(temp_deno.^2,1); % prob=mean2(prob);
Prob= min(max(Prob,0),1); % Projection to the convex set
end