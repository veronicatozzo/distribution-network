function [ phix ] = eval_basis( x, inds, varargin )
%EVAL_BASIS Summary of this function goes here
%   Detailed explanation goes here
[n, d] = size(x);
if isempty(varargin)
    do_cos_basis = true;
else
    do_cos_basis = false;
end
% set the one-d basis
if ~do_cos_basis
    phi_k = @(x,k) (k==0)+(k<0)*sqrt(2)*cos(2*pi*k*x)+(k>0)*sqrt(2)*sin(2*pi*k*x);
else
    phi_k = @(x,k) (k==0)+(k>0)*sqrt(2)*cos(pi*k*x);
end
phix = ones(n,size(inds,1));
for di=1:d
    % evaluate one-d basis functions
    mav = max(inds(:,di));
    miv = min(inds(:,di));
    inds_di = miv:mav;
    
    if ~do_cos_basis
        phidi = ones(n,length(inds_di));
        for k=1:length(inds_di)
            phidi(:,k) = phi_k(x(:,di),inds_di(k));
        end
    else
        phidi = sqrt(2)*cos(pi*bsxfun(@times,x(:,di),inds_di));
        phidi(:,inds_di==0) = 1;
    end
    
    % multiply with the one-d basis functions (indexing starting from 1)
    phix = phix.*phidi(:,inds(:,di)-miv+1);
end

end

