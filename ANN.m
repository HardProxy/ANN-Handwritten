# Funções necessárias para a aplicação da rede neural.
#
# Medida de Erro : MSE ( Erro quadratico ).
# Função de ativação : sigmoid ( Classificação ).
#
# Para o input @var{x} temos que as linhas serão iguais ao numero de features e 
# suas colunas serão o tamanho do data sample. Para o label @var{t} teremos as 
# linhas sendo o numero de outputs(classificações) e as colunas serão os outputs
# relacionado ao data sample.
#
# Os pesos da matriz a @var{a} terão o numero de linhas como o numero de Hidden 
# nodes sem o termo de bias e a o numero de colunas serão o numero de features 
# mais o termo de bias.
#
# Os pesos da matriz b @var{b} terao o numero de linhas como o numero de outputs
# sem o termo de Bias e seu numero de colunas serão o numero de hidden nodes mais
# o nodo de Bias.
#
#  As épocas @var{epoques} será a quantidade de vezes que a rede será treinada  
# dentro de um mesmo data sample. O Epsilon @var{epsilon} será o parâmetro de 
# Learning Rate.
# 
# O programa salva o erro a cada época para verificação da convergência dos
# resultados.
#
# Otaviano Cruz
# email : otavianocruz@id.uff.br
#
#

function [a,b] = ANN(x,t,h_Nodes, epoques, epsilon )
  
 
  [inputs,samples] = size(x);
  [outputs] = size(t,1);
  
  diff_E_b = [];
  diff_E_a = [];
  a = rand( h_Nodes , inputs + 1 );
  b = rand( outputs , h_Nodes + 1);
  
  
  for j = 1:epoques,
   for i = 1:samples,
     
    % Forward Prop
    
    u = a * [1; x(:,i)];
    y = sigmoid( u );
    v = b * [1; y];
    z = sigmoid( v );
  
    % Error measure
    Error = MSE(z,t(:,i));

    % Back Prop
    p = (z - t(:,i)) .* z .* ( 1 - z );
    diff_E_b = [ diff_E_b;  p * [1; y]' ];  
  
    q = b( : , 2 : end )' *p .* y .* ( 1 - y );
    diff_E_a =[ diff_E_a; q * [1; x(:,i)]' ] ;
 
    % Atualização de Pesos (Conferir valores)   
       
   endfor;
   d_a = sum(diff_E_a);
   d_b = sum(diff_E_b);
   diff_E_a = [];
   diff_E_b = [];
   a += -epsilon.* d_a;
   b += -epsilon.* d_b;
    
   u = a * [ ones(1, samples); x];
   y = sigmoid( u );
   v = b * [ ones(1,samples ); y];
   z = sigmoid( v );
  endfor;  
endfunction;
