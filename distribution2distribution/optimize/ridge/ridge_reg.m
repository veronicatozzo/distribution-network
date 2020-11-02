function [rreg, Yhat, SM] = ridge_reg(X, Y, varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

lambdars = get_opt(opts,'lambdars',2.^(20:-1:-20));
cv = get_opt(opts,'cv','loo');
if strcmp(cv,'none') && length(lambdars)>1
    cv = 'loo';
end
rreg.cv.lambdars = lambdars;
eigen_decomp = get_opt(opts,'eigen_decomp',true);
verbose = get_opt(opts, 'verbose', false);
rstats = get_opt(opts, 'rstats');
eval_beta = get_opt(opts, 'eval_beta', true);


[n,~] = size(X);
nlams = length(lambdars);
lam_mse = nan(nlams,1);
best_mse = inf;
best_beta = nan;
switch cv
    case 'loo' % TODO: implement loo predictions
        if isempty(rstats)
            rstats = rreg_stats(X,Y,eigen_decomp);
        end
        for li=1:nlams
            beta = rreg_beta(rstats, lambdars(li));
            Yhat = X*beta;
            SMii = rreg_SMii(X, rstats, lambdars(li));
            lam_mse(li) = mean( sum(bsxfun(@times,Y-Yhat,1./(1-SMii)).^2,2) );
            if lam_mse(li)<best_mse
                best_mse = lam_mse(li);
                best_beta = beta;
            end
        end
        [~,li] = min(lam_mse);
        rreg.cv.lam_mse = lam_mse;
        
    case 'hold' 
        trn_set = get_opt(opts,'trn_set');
        trn_perc = get_opt(opts,'trn_perc',.8);
        if isempty(trn_set)
            trn_set = false(n,1);
            trn_set(randperm(n,ceil(trn_perc*n))) = true;
        end
        if isempty(rstats)
            rstats = rreg_stats(X(trn_set,:),Y(trn_set,:),eigen_decomp);
        end
        for li=1:nlams
            beta = rreg_beta(rstats, lambdars(li));
            Yhat = X(~trn_set,:)*beta;
            lam_mse(li) = mean( sum((Y(~trn_set,:)-Yhat).^2,2) );
            if lam_mse(li)<best_mse
                best_mse = lam_mse(li);
                best_beta = beta;
            end
        end
        [~,li] = min(lam_mse);
        rstats = rreg_stats(X,Y,eigen_decomp);
        rreg.cv.lam_mse = lam_mse;
        rreg.cv.trn_set = trn_set;
        
    case 'folds' % TODO: implement
        
        
    otherwise
        li=1;
        
end

if verbose
    fprintf('\tlam_i: (%i,%f), {%f}\n', li, lambdars(li), min(lam_mse(:)) );
end

if eval_beta
    beta = rreg_beta(rstats, lambdars(li));
else
    beta = best_beta;
end
Yhat = X*beta;
rreg.lambda = lambdars(li);
rreg.beta = beta;
if nargout>2
    SM = rreg_SM(X, rstats, lambdars(li));
end

end
