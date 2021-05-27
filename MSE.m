function E = MSE(pred, out)
  [K,N] = size(pred);
  
  E = (1./(2.0 .* N .* K )) .* ( out - pred ).**2;
  
endfunction;