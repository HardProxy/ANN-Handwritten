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
#
# [a,b] = ANN(x,t,h_Nodes,epoques,epsilon)
#
# Otaviano Cruz
# email : otavianocruz@id.uff.br
#
function main

  load "data_Hand.mat";
  
  m = size(X, 1);
  num_train = round(m * .7); # 70 por cento dos dados sao dedicados ao treino
  X_train = X(1:num_train, :);
  X_test = X(num_train+1:end, :);
  
  
  
  y_multiclass = [y==1 y==2 y==3 y==4 y==5 y==6 y==7 y==8 y==9 y==10];
  y_train = y_multiclass(1:num_train, :);
  
  h_Nodes = 256;
  epoques = 100000;
  epsilon = 0.12;
  [a,b] = ANN(X',y',h_Nodes,epoques,epsilon);
  
  u = [ ones(m,1) X_train ] * a;
  y1 = sigmoid( u ); 
  v = [ ones(m,1) y1 ] * b;
  z = sigmoid( v );
  
  
  [probability, pred_number] = max(z');
  pred_errors = pred_number != shuffled_y(num_train+1:end, :);
  num_test_observations = size(X_test, 1);
  num_test_errors = sum(pred_errors);
  num_test_hits = num_test_observations - num_test_errors;
  accuracy = 100.0 * num_test_hits / num_test_observations;
  printf("\n\n*** The model was %5.2f%% accurate against the test set. ***\n", accuracy);
endfunction;
