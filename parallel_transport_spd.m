function zeta = parallel_transport_spd(X, Y, eta)
E = sqrtm((Y/X));
zeta = E*eta*E';

end