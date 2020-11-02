function [Y0_est] = predict(Pc,rks, B)
    z0 = rks.rfeats(Pc);
    Y0_est = z0*B;
end

