function Y = retraction_spd(X, eta, t)
teta = t*eta;
symm = @(X) .5*(X+X');
Y = symm(X + teta + .5*teta*(X\teta));
[~,index] =chol(Y);
iter = 1; max_iter = 5;
while (index)&&(iter<=max_iter)
    iter = iter+1;
    t = t/2;
    teta = t*eta;
    Y = symm(X + teta + .5*teta*(X\teta));
    [~,index] =chol(Y);
end   
if iter>=max_iter
    Y=X;
end

end
