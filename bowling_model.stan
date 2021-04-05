
data {
  
  int<lower=0> N; // Number of innings in total
  int<lower=0> P; // Number of players in total
  //int<lower=0> Y; // Number of years in data set
  int<lower=0> D; // Number of decades in data set
  int<lower=0> I; // Number of innings in data set, usually 3
  int<lower=0> O; // Number of opposition teams in data set
  int<lower=0> OE; // Number of opposition eras in data set
  
  matrix<lower=0, upper=1>[N, P] Players; // Players data
  
  // matrix<lower=0, upper=1>[N, Y] Year; // Years data
  
  matrix<lower=0, upper=1>[N, D] Decade; // Decades data
  
  vector[N] Age; // Age data
  vector[N] Age_squared; // Squared age data
  
  vector<lower=0, upper=1>[N] HA; // Home/away effect data
  
  matrix<lower=0, upper=1>[N, I] Inns; // Innings number data
  matrix<lower=0, upper=1>[N, O] Opposition; // Opposition data
  matrix<lower=0, upper=1>[N, OE] OppEra; // Opposition Era data
  
  int<lower=0> Runs[N]; // Runs scored
  
}

transformed data {
  
  vector[N] Age_std = (Age - mean(Age))/sd(Age);
  vector[N] Age_squared_std = (Age_squared - mean(Age_squared))/sd(Age_squared);
  
}

parameters {
  
  vector[P] mu_theta; // mean for ability prior
  vector<lower=0>[P] sigma2_theta; // variance for ability prior
  vector[P] theta_tilde; // dummy variable

  vector[D] epsilon; //Dummy variable for random walk prior
  vector<lower=0>[D] sigma2_delta; // variance for year effect prior
  
  vector[P] alpha1; // Peak age for each player
  vector[P] alpha2; // Ageing curvature for each player
  
  real zeta; // Home/away effect parameter
  
  vector[I] nu; // Innings effect parameter
  vector[O] xi; // Opposition effect parameter
  vector[OE] omega; // Opposition/Era interaction parameter
  
}
transformed parameters {
  
  vector[P] theta; // transform of theta_tilde
  vector[D] delta; // Decade effect parameter
  
  vector[N] lambda; // mean for Poisson
  
  vector[N] alpha1_vector = Players * alpha1; // alpha_1 repeated the correct number of times for each player
  vector[N] alpha2_vector= Players * exp(alpha2); // alpha_2 repeated the correct number of times for each player
  
  for(i in 1:P){
    theta[i] = mu_theta[i] + sqrt(sigma2_theta[i])*theta_tilde[i];
  }
  
  delta[D] = sqrt(sigma2_delta[D])*epsilon[D];
  for(l in 1:(D-1)) {
    delta[(D-l)] = delta[((D+1)-l)] + sqrt(sigma2_delta[(D-l)])*epsilon[(D-l)];
  }
  
  lambda = exp((Players*theta) + 
               (zeta*HA) + 
               ((-1 * alpha2_vector) .* ((Age_squared_std) - (alpha1_vector .* (Age_std)) + (alpha1_vector .* alpha1_vector))) + 
               (Inns*nu) + 
               (Opposition*xi) + 
               (Decade*delta) + 
               (OppEra*omega));
}

model {
  
  // priors
  
  mu_theta ~ normal(log(45), 0.25);
  sigma2_theta ~ inv_gamma(3, 1);
  theta_tilde ~ std_normal();
  
  sigma2_delta ~ inv_gamma(2, 100); // Stan uses a=shape, b=scale whereas the Boys/Philipson have b=rate
  
  epsilon ~ std_normal();
  
  zeta ~ normal(0, 0.5);
  nu ~ normal(0, 0.5);
  xi ~ normal(0, 0.5);
  omega ~ normal(0, 0.5);
  
  alpha1 ~ std_normal();
  alpha2 ~ std_normal();

  // model
  
  Runs ~ poisson(lambda);

}
