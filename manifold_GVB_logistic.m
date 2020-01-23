function [mu,Sig,LB_smooth] = manifold_GVB_logistic(X_train,y_train)
% fitting a logistic regression using the Manifold Gaussian VB algorithm
% INPUT: 
%   X_train:        design matrix
%   y_train:        binary response vector
% OUTPUT:
%   mu:             mean mu of the Gaussian VB
%   Sig:            covariance matrix of the Gaussian VB
%   LB_smooth:      lower bound (after being smoothed by a moving average)
% Reference: "Variational Bayes on Manifold" by Tran et al.
% @Written by Minh-Ngoc Tran
% ================================================================%

hp_sig2 = 50; % hyperparamter. The prior for coefficient beta is N(0,hp_sig2 I)
d = size(X_train,2);
patience_parameter = 10; % stop updating if lower bound isn't improved after patience_parameter iterations
smooth_window = 10; % window size to smoothen lower bound 
max_iter = 2000; % maximum number of iteration in the VB updating
stepsize_threshold = max_iter/2; % from this iteration, stepsize is reduced 

S = 100; % number of samples used to estimate the LB gradient 
eps0 = .01; % learning rate
momentum_weight = 0.4; % weight in the momentum method; should be around 0.3-0.5 based on experience
mu = zeros(d,1); 
Sig = .04*eye(d);
c12 = zeros(1,d+d*d); % control variate, initilised to be all zero
gra_log_q_lambda = zeros(S,d+d*d); % gradient of log_q
grad_log_q_h_function = zeros(S,d+d*d); % (gradient of log_q) x h(theta) 
grad_log_q_h_function_cv = zeros(S,d+d*d); % control_variate version: (gradient of log_q) x (h(theta)-c)
Sig_inv = eye(d)/Sig;
rqmc = normrnd_qmc(S,d);      % generate standard normal numbers, using quasi-MC
C_lower = chol(Sig,'lower');
parfor s = 1:S        
    beta = mu+C_lower*rqmc(s,:)';
    llh = log_likelihood_logistic(y_train,X_train,beta);

    prior_beta = -d/2*log(2*pi)-d/2*log(hp_sig2)-beta'*beta/hp_sig2/2;
    log_q_lambda = -d/2*log(2*pi)-1/2*log(det(Sig))-1/2*(beta-mu)'*Sig_inv*(beta-mu);
    h_function = prior_beta+llh-log_q_lambda;
    
    aux = Sig_inv*(beta-mu);
    gra_log_q_mu = aux;
    gra_log_q_Sig = -1/2*Sig_inv+1/2*aux*(aux');    
    gra_log_q_lambda(s,:) = [gra_log_q_mu;gra_log_q_Sig(:)]';
    grad_log_q_h_function(s,:) = gra_log_q_lambda(s,:)*h_function;    
    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_function-c12);
end
c12 = zeros(1,d+d*d); 
for i = 1:d+d*d
    aa = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
    c12(i) = aa(1,2)/aa(2,2);
end
Y12 = mean(grad_log_q_h_function_cv)'; % Euclidiance gradient of lower bounf LB
% To use manifold GVB for other models, all we need is Euclidiance gradient
% of LB. All the other stuff below are model-independent.
gradLB_mu = Sig*Y12(1:d);% natural gradient of LB w.r.t. mu
gradLB_Sig = Sig*reshape(Y12(d+1:end),d,d)*Sig; % natural gradient of LB w.r.t. Sigma
gradLB_Sig_momentum = gradLB_Sig; % initialise momentum gradient for Sig
gradLB_mu_momentum = gradLB_mu;% initialise momentum gradient for Sig

mu_best = mu; Sig_best = Sig;
iter = 0; stop = false; LB = 0; patience = 0; LB_smooth = 0;
while ~stop    
    iter = iter+1    
    if iter>stepsize_threshold
        stepsize=eps0*stepsize_threshold/iter;
    else
        stepsize=eps0;
    end    
    Sig_old = Sig;    
    Sig = retraction_spd(Sig_old,gradLB_Sig_momentum,stepsize); % retraction to update Sigma
    mu = mu+stepsize*gradLB_mu_momentum; % update mu
    
    gra_log_q_lambda = zeros(S,d+d*d); 
    grad_log_q_h_function = zeros(S,d+d*d); 
    grad_log_q_h_function_cv = zeros(S,d+d*d); % control_variate
    log_llh = zeros(S,1);
    Sig_inv = eye(d)/Sig;
    rqmc = normrnd_qmc(S,d);      
    C_lower = chol(Sig,'lower');
    parfor s = 1:S    
        beta = mu+C_lower*rqmc(s,:)';
        llh = log_likelihood_logistic(y_train,X_train,beta);
        log_llh(s) = llh;

        prior_beta = -d/2*log(2*pi)-d/2*log(hp_sig2)-beta'*beta/hp_sig2/2;
        log_q_lambda = -d/2*log(2*pi)-1/2*log(det(Sig))-1/2*(beta-mu)'*Sig_inv*(beta-mu);
        h_function = prior_beta+llh-log_q_lambda;

        aux = Sig_inv*(beta-mu);
        gra_log_q_mu = aux;
        gra_log_q_Sig = -1/2*Sig_inv+1/2*aux*(aux');    
        gra_log_q_lambda(s,:) = [gra_log_q_mu;gra_log_q_Sig(:)]';
        grad_log_q_h_function(s,:) = gra_log_q_lambda(s,:)*h_function;    
        grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_function-c12);
    end
    for i = 1:d+d*d
        aa = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
        c12(i) = aa(1,2)/aa(2,2);
    end  
    Y12 = mean(grad_log_q_h_function_cv)';
    % clipping the gradient
    grad_norm = norm(Y12);
    norm_gradient_threshold = 100;
    if grad_norm>norm_gradient_threshold
        Y12 = (norm_gradient_threshold/grad_norm)*Y12;
    end

    gradLB_mu = Sig*Y12(1:d);
    gradLB_Sig = Sig*reshape(Y12(d+1:end),d,d)*Sig;
    
    zeta = parallel_transport_spd(Sig_old,Sig,gradLB_Sig_momentum); % vector transport to move gradLB_Sig_momentum
    % from previous Sig_old to new point Sigma
    gradLB_Sig_momentum = momentum_weight*zeta+(1-momentum_weight)*gradLB_Sig; % update momentum grad for Sigma
    gradLB_mu_momentum = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu; % update momentum grad for mu
    
    % lower bound
    LB(iter) = -d/2*log(hp_sig2)-(trace(Sig)+mu'*mu)/hp_sig2/2+log(det(Sig))/2+d/2+mean(log_llh);
    if iter>smooth_window
        LB_smooth(iter-smooth_window) = mean(LB(iter-smooth_window:iter));    % smooth out LB by moving average
        LB_current = LB_smooth(end);
        if LB_smooth(iter-smooth_window)>=max(LB_smooth)
            mu_best = mu; Sig_best = Sig;
            patience = 0;
        else
            patience = patience+1;
        end
    end
    if (patience>patience_parameter)||(iter>max_iter) stop = true; end     
    
end
mu = mu_best; Sig = Sig_best;
end

