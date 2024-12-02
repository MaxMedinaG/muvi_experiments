import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam
import torch
from torch.distributions import constraints


class PPCA(PyroModule):
    def __init__(self, n_components, data_dim, device='cpu'):
        """Base class for Probabilistic PCA (PPCA)."""
        super().__init__()
        self.n_components = n_components
        self.data_dim = data_dim
        self.device = device

    def fit(self, data, num_steps=1000, **optim_kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    def transform(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def reconstruct(self, latent):
        """Reconstructs data from latent representations."""
        latent = latent.to(self.device)
        W = self.get_W()
        return torch.matmul(latent, W.T)

    def get_W(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_parameters(self):
        raise NotImplementedError("Subclasses should implement this method.")


class PPCA_VI(PPCA):
    def __init__(self, n_components, data_dim, device='cpu'):
        super().__init__(n_components, data_dim, device)
        # Define prior distributions for W and noise_variance
        self.W_prior = dist.Normal(
            torch.zeros(data_dim, n_components, device=device),
            torch.ones(data_dim, n_components, device=device)
        ).to_event(2)
        self.noise_var_prior = dist.InverseGamma(
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device)
        )

    def model(self, data):
        N = data.size(0)
        W = pyro.sample("W", self.W_prior)
        noise_var = pyro.sample("noise_variance", self.noise_var_prior)
        with pyro.plate("observations", N):
            z = pyro.sample(
                "z",
                dist.Normal(
                    torch.zeros(N, self.n_components, device=self.device),
                    torch.ones(N, self.n_components, device=self.device)
                ).to_event(1)
            )
            mean = torch.matmul(z, W.T)
            pyro.sample(
                "obs",
                dist.Normal(mean, noise_var.sqrt()).to_event(1),
                obs=data
            )

    def guide(self, data):
        N = data.size(0)
        D, K = self.data_dim, self.n_components
        q_W_loc = pyro.param(
            "q_W_loc",
            torch.zeros(D, K, device=self.device)
        )
        q_W_scale = pyro.param(
            "q_W_scale",
            0.1 * torch.ones(D, K, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample("W", dist.Normal(q_W_loc, q_W_scale).to_event(2))
        q_log_noise_var_loc = pyro.param(
            "q_log_noise_var_loc",
            torch.tensor(0.0, device=self.device)
        )
        q_log_noise_var_scale = pyro.param(
            "q_log_noise_var_scale",
            torch.tensor(0.1, device=self.device),
            constraint=constraints.positive
        )
        pyro.sample(
            "noise_variance",
            dist.LogNormal(q_log_noise_var_loc, q_log_noise_var_scale)
        )
        with pyro.plate("observations", N):
            q_z_loc = pyro.param(
                "q_z_loc",
                0.01 * torch.randn(N, K, device=self.device)
            )
            q_z_scale = pyro.param(
                "q_z_scale",
                0.1 * torch.ones(N, K, device=self.device),
                constraint=constraints.positive
            )
            pyro.sample("z", dist.Normal(q_z_loc, q_z_scale).to_event(1))

    def fit(self, data, num_steps=1000, **optim_kwargs):
        data = data.to(self.device)
        pyro.clear_param_store()
        optimizer = pyro.optim.ClippedAdam(optim_kwargs)
        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=pyro.infer.Trace_ELBO()
        )
        losses = []
        for step in range(num_steps):
            loss = svi.step(data)
            losses.append(loss)
            if step % 500 == 0:
                print(f"Step {step} - Loss: {loss:.4f}")
        return losses

    def transform(self, data):
        data = data.to(self.device)
        self.guide(data)
        return pyro.param("q_z_loc").detach()

    def get_W(self):
        return pyro.param("q_W_loc").detach()

    def get_parameters(self):
        W = self.get_W()
        q_log_noise_var_loc = pyro.param("q_log_noise_var_loc").detach()
        q_log_noise_var_scale = pyro.param("q_log_noise_var_scale").detach()
        noise_variance = torch.exp(
            q_log_noise_var_loc + 0.5 * q_log_noise_var_scale ** 2
        )
        return {"W": W, "noise_variance": noise_variance}


class PPCA_MAP(PPCA):
    def __init__(self, n_components, data_dim, device='cpu'):
        super().__init__(n_components, data_dim, device)
        self.W = torch.nn.Parameter(
            torch.randn(data_dim, n_components, device=device)
        )
        self.log_noise_variance = torch.nn.Parameter(
            torch.tensor(0.0, device=device)
        )

    def fit(self, data, num_steps=1000, **optim_kwargs):
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), **optim_kwargs)
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            loss = self._map_loss(data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % 500 == 0:
                print(f"Step {step} - Loss: {loss.item():.4f}")
        return losses

    def _map_loss(self, data):
        N, D = data.shape
        K = self.n_components
        W = self.W  # Shape: (D, K)
        noise_var = torch.exp(self.log_noise_variance)
        # z = self.z  # Shape: (N, K), initialize and optimize this
        # Priors
        W_prior = dist.Normal(
            torch.zeros_like(W),
            torch.ones_like(W)
        )
        log_prior_W = W_prior.log_prob(W).sum()

        noise_var_prior = dist.InverseGamma(
            torch.tensor(1.0, device=self.device),
            torch.tensor(1.0, device=self.device)
        )
        log_prior_noise_var = noise_var_prior.log_prob(noise_var)

        # Likelihood
        M = W @ W.T + noise_var * torch.eye(D, device=self.device)
        mean_zero = torch.zeros(D, device=self.device)
        mvn = dist.MultivariateNormal(mean_zero, covariance_matrix=M)
        log_likelihood = mvn.log_prob(data).sum()

        # Negative log-posterior
        return - (log_likelihood + log_prior_W + log_prior_noise_var)

    def transform(self, data):
        data = data.to(self.device)
        W = self.W.detach()
        noise_var = torch.exp(self.log_noise_variance.detach())
        M = W.T @ W + noise_var * torch.eye(self.n_components, device=self.device)
        M_inv = torch.inverse(M)
        return (M_inv @ W.T @ data.T).T

    def get_W(self):
        return self.W.detach()

    def get_parameters(self):
        W = self.get_W()
        noise_variance = torch.exp(self.log_noise_variance.detach())
        return {"W": W, "noise_variance": noise_variance}
