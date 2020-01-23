function llh = log_likelihood_logistic(y,X,beta)
% compute log-likelihood for the logistic regression model
aux = X*beta;
llh = y.*aux-log(1+exp(aux));
llh = sum(llh);

end