import torch
import torch.nn as nn
from torch.autograd import Variable



class SplineRegression(nn.Module):
    def __init__(self, n_nodes, n_groups):
        super(SplineRegression, self).__init__()
        
        self.n_nodes = n_nodes
        self.n_groups = n_groups
        self.alphas = nn.Embedding(n_groups, n_nodes)
        
    def forward(self, t, c):
        alphas = self.alphas(c)
        return torch.bmm(t.unsqueeze(1), alphas.unsqueeze(2)).squeeze()

class BayesianSplineRegression(nn.Module):
    def __init__(self, n_nodes, n_groups):
        super(BayesianSplineRegression, self).__init__()
        self.regression = SplineRegression(n_nodes, n_groups)
        self.n_nodes = n_nodes        
        self.n_groups = n_groups        
        pyro.clear_param_store()
        
    def _set_w_reg(self, w):
        self.regression.alphas.weight.data = w
    
    def sample(self, t, c, n=1):
        X = np.empty((n, len(t)))
        for i in range(n):
            w_dict = self.guide()
            self._set_w_reg(w_dict['module$$$alphas.weight'])
            y = self.regression(t, c).detach().cpu().numpy()
            X[i, :] = y.ravel()
        return X
    
    def model(self, t, y, c):
        n_groups, n_nodes = self.n_groups, self.n_nodes
        
#         w_prior = dist.Normal(torch.zeros(n_groups, n_nodes), torch.ones(n_groups, n_nodes)).to_event(1)
        scale_a = pyro.sample("sigma_a", dist.HalfCauchy(5.*torch.ones(n_groups)))
        w_prior = GaussianRandomWalk(scale_a, n_nodes).to_event(1)
        
        priors = {'alphas.weight': w_prior}
        scale = pyro.sample("sigma", dist.HalfCauchy(5.))
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.regression, priors)
        # sample a nn (which also samples w and b)
        lifted_reg_model = lifted_module()
        with pyro.plate("map", len(t)):
            # run the nn forward on data
            prediction_mean = lifted_reg_model(t, c).squeeze(-1)
            # condition on the observed data
            pyro.sample("obs",
                        dist.Normal(prediction_mean, scale),
                        obs=y)
            return prediction_mean
    
    def fit(self, c, t, y, lr=.001, n_iter=10):
        pyro.clear_param_store()
        
        self.guide = AutoDiagonalNormal(self.model)
        optim = Adam({"lr": lr})
        elbo = Trace_ELBO()
        self.svi = SVI(self.model, self.guide, optim, loss=elbo)
        losses = []
        for j in range(n_iter):
            loss = self.svi.step(t, y, c)
            if j % 250 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(t)))
            losses.append(loss / len(t))
            
        return losses
