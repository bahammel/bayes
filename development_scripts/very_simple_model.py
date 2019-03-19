def create_data(mu, sd):
	data = torch.zeros(500)
	for i in range(500):
		data[i] = torch.distributions.Normal(mu, sd).sample()
	return data

data = create_data(10, 3)

def model(data):
	mu0 = pyro.sample("latent_mu0", dist.Normal(0.0, 1.0))
	sd0 = pyro.sample("latent_sd0", dist.Gamma(1.0, 1.0))

	with pyro.plate("observed_data"):
		pyro.sample("obs", dist.Normal(mu0, sd0), obs=data)

def guide(data):
	mu0_mu_q = pyro.param("mu0_mu_q", torch.tensor(0.0), constraint=constraints.real)
	mu0_sd_q = pyro.param("mu0_sd_q", torch.tensor(1.0), constraint=constraints.positive)
	sd0_alpha_q = pyro.param("sd0_alpha_q", torch.tensor(1.0), constraint=constraints.positive)
	sd0_beta_q = pyro.param("sd0_beta_q", torch.tensor(1.0), constraint=constraints.positive)

	pyro.sample("latent_mu0", dist.Normal(mu0_mu_q, mu0_sd_q))
	pyro.sample("latent_sd0", dist.Gamma(sd0_alpha_q, sd0_beta_q))

def train(n_steps, data):
	pyro.clear_param_store()

	adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
	optimizer = Adam(adam_params)
	svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

	for _ in range(n_steps):
		svi.step(data)
