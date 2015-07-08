ROCK* - the algorithm in this program combines Gaussian kernel regression and natural gradient descent and is shown to have better convergence than FDSA, SPSA, IW-PGPE, REINFORCE, CMAES, AMalGaM, and CEM: 

Jemin Hwangbo, Christian Gehring, Hannes Sommer, Roland Siegwart and Jonas Buchli.2014. ROCK* - Efficient Black-box Optimization for Policy Learning. Humanoids 2014, Madrid, Spain

The natural gradient is inspired from here:

S. Amari, and S.C. Douglas. 1998. Why natural gradient? IEEE 1213-1216

and the covariant matrix adaptation (CMA-ES-like) is from here:

Nikolas Hansen and Andreas Ostermeier. 1996. Adapting arbitrary normal mutation distributions in evolution strategies: the covariance matrix adaptation. IEEE 312-317

The objective function is like the one presented in:

https://chessprogramming.wikispaces.com/Texel%27s+Tuning+Method
