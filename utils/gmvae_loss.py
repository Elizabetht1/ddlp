import torch

def calc_gmvae_loss(y, 
    mc_w_samples,
    mc_x_samples,
    mu_x, 
    logvar_x,
    decoder_beta,
    M=10,
    K=5):

    """
    GMVAE Loss Function based on Dilokthanakul et al. (2017)
    
    Args:
        y : observed sample 
        mu_x, logvar_X : q_phi_x(x|y) parameters
        mu_w, logvar_w : q_phi_w(w|y) parameters
        M: number of Monte Carlo samples
        K: number of Gaussian mixture components
    """
    
    
    batch_size = y.size(0)
    
    # Initialize loss components
    # reconstruction_loss = 0
    conditional_prior_loss = 0
    
    # Monte Carlo sampling loop
    # Sample using reparameterization trick
    # mc_w_samples = reparameterize(mu_w.repeat(M*2,dim=0), logvar_w.repeat(M*2,dim=0),)  # [batch_size * M, dim_w]
    # w_sample1, w_sample2 = torch.chunk(mc_w_samples,chunks=2,dim=0)
    # dim_w = w_sample1.shape[-1]
    # mc_x_samples = reparameterize(mu_x.repeat(M*2,dim=0), logvar_x.repeat(M*2,dim=0))  # [batch_size * M, dim_x]
    x_sample1, x_sample2 = torch.chunk(mc_x_samples,chunks=2,dim=0)
    dim_x = x_sample1.shape[-1]


    # === CONDITIONAL PRIOR TERM (Equation 5) ===
    # E_q(w|y)p(z|x,w)[KL(q_phi_x(x|y) || p_beta(x|w,z))]
        
    # Get mixture component parameters from decoder_beta
    # decoder_beta outputs K means and K variances
    mixture_params = decoder_beta(mc_w_samples).view(-1,batch_size, 2*K*mc_x_samples.shape[-1]) # [batch_size*M*2, 2*K*dim_x]
    mixture_params_cp,mixture_params_zp = torch.chunk(mixture_params,chunks=2,dim=0) # [M, batch_size, 2*K*dim_x]

    mu_mixture_cp = mixture_params_cp[:, :K*dim_x].view(batch_size*M, K, dim_x)  # [batch_size*M, K, dim_x]
    logvar_mixture_cp = mixture_params_cp[:, K*dim_x:].view(batch_size*M, K, dim_x)  # [batch_size*M, K, dim_x]
    
    # Compute p_beta(z_k=1|x,w) using Equation (3)
    # p(z_k=1|x,w) = π_k * N(x|μ_k,σ_k) / Σ_j π_j * N(x|μ_j,σ_j)
    weight_k = compute_mixture_posterior(x_sample1, mu_mixture_cp, logvar_mixture_cp)  # [batch_size*M,K]
        
    
    conditional_prior_sample = []
        
    for k in range(K):
        # Extract parameters for k-th mixture component
        mu_k = mixture_params_cp[:, k, :]      # [batch_size*M, dim_x]
        logvar_k = logvar_mixture_cp[:, k, :]  # [batch_size*M, dim_x]
            
        # Compute KL divergence: KL[q_phi_x(x|y) || p_beta(x|w,z_k=1)]
        kl_qx_px_k = kl_divergence_gaussians(mu_x, logvar_x, mu_k, logvar_k)  # [batch_size*M]
        conditional_prior_sample.append(kl_qx_px_k)

    conditional_prior_sample = torch.stack(conditional_prior_sample,dim=1) # [batch_size*M, K]
    conditional_prior_sample = conditional_prior_sample.view(-1,M,K)
    # Weight the KL divergence
    conditional_prior_sample = (weight_k[None,:,:] * conditional_prior_sample)#[batch_size, M, K]

    
    # Average over Monte Carlo samples
    conditional_prior_loss = conditional_prior_sample.sum() / (M * batch_size)  # Average over batch too
    
    
    # === Z-PRIOR TERM (Equation 6) ===
    # -E_q(x|y)q(w|y)[KL(p_beta(z|x,w) || p(z))] where p(z) = Uniform
    z_prior_loss = 0
    mu_mixture_zp = mixture_params_zp[:, :K*dim_x].view(M* batch_size, K, dim_x)
    logvar_mixture_zp = mixture_params_zp[:, K*dim_x:].view(M*batch_size, K, dim_x)
    
    # Compute all mixture posterior probabilities p(z_k|x,w)
    z_posterior_probs = compute_all_mixture_posteriors(x_sample2, mu_mixture_zp, logvar_mixture_zp, K)  # [batch_size, K]
        

    # KL divergence with uniform distribution (entropy of categorical)
    # KL[p(z|x,w) || Uniform] = -H[p(z|x,w)] + log(K)
    uniform_prior = torch.ones_like(z_posterior_probs) / K
    kl_z = torch.sum(z_posterior_probs * (torch.log(z_posterior_probs + 1e-8) - torch.log(uniform_prior)), dim=1)
    kl_z /= (M * batch_size)
    
    # === OPTIONAL: MINIMUM INFORMATION CONSTRAINT ===
    # As per Section 3.4, apply constraint to z-prior term if needed
    # L'_z = -max(λ, E[KL(p(z|x,w)||p(z))])
    # Uncomment below if using the constraint:
    # lambda_threshold = 1.0  # Set based on your needs
    # z_prior_loss = torch.max(torch.tensor(lambda_threshold), z_prior_loss)
    
    # Total loss (note the signs - we minimize negative ELBO)
    # total_loss = -reconstruction_loss + conditional_prior_loss - kl_z
    
    return conditional_prior_loss, z_prior_loss


def reparameterize(mu, logvar):
    """Reparameterization trick"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_gaussians(mu1, logvar1, mu2, logvar2):
    """
    KL divergence between two Gaussians: KL[N(mu1,sigma1) || N(mu2,sigma2)]
    Returns: [batch_size] tensor
    """
    kl = 0.5 * (logvar2 - logvar1 + 
                torch.exp(logvar1 - logvar2) + 
                (mu1 - mu2).pow(2) * torch.exp(-logvar2) - 1)
    return kl.sum(dim=1)  # Sum over dimensions, keep batch dimension


def kl_divergence_standard_normal(mu, logvar):
    """KL divergence with standard normal N(0,I)"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def compute_mixture_posterior(x, mu_mixture, logvar_mixture, k, K):
    """
    Compute p_beta(z_k=1|x,w) using Equation (3)
    
    Args:
        x: [batch_size, M, dim_x]
        mu_mixture: [batch_size, M, K, dim_x] 
        logvar_mixture: [batch_size, M, K, dim_x]
        k: component index
        K: total number of components
    
    Returns:
        [batch_size] tensor of posterior probabilities
    """
    batch_size = x.size(0)
    
    # Compute log probabilities for all components
    log_probs = torch.zeros(batch_size, K, device=x.device)
    
    for j in range(K):
        mu_j = mu_mixture[:, j, :]      # [batch_size, dim_x]
        logvar_j = logvar_mixture[:, j, :]  # [batch_size, dim_x]
        
        # Log probability of x under component j: log N(x|mu_j, exp(logvar_j))
        log_prob_j = -0.5 * (logvar_j + (x - mu_j).pow(2) * torch.exp(-logvar_j))
        log_probs[:, j] = log_prob_j.sum(dim=1)  # Sum over dimensions
    
    # Add uniform prior log probabilities (log(1/K) for each component)
    log_probs += torch.log(torch.tensor(1.0 / K))
    
    # Compute posterior probabilities using log-sum-exp trick
    log_prob_k = log_probs[:, k]
    log_sum_all = torch.logsumexp(log_probs, dim=1)
    
    return torch.exp(log_prob_k - log_sum_all)


def compute_all_mixture_posteriors(x, mu_mixture, logvar_mixture, K):
    """
    Compute p_beta(z_k=1|x,w) for all k
    
    Returns:
        [batch_size, K] tensor of posterior probabilities
    """
    batch_size = x.size(0)
    log_probs = torch.zeros(batch_size, K, device=x.device)
    
    # for j in range(K):
    #     mu_j = mu_mixture[:, j, :]
    #     logvar_j = logvar_mixture[:, j, :]
    #     log_prob_j = -0.5 * (logvar_j + (x - mu_j).pow(2) * torch.exp(-logvar_j))
    #     log_probs[:, j] = log_prob_j.sum(dim=1)

    log_probs = -0.5 * (logvar_mixture + (x - mu_mixture).pow(2) * torch.exp(-logvar_mixture)).sum(dim=1)
    
    # Add uniform prior
    log_probs += torch.log(torch.tensor(1.0 / K))
    
    # Convert to probabilities
    log_sum_all = torch.logsumexp(log_probs, dim=1, keepdim=True)
    return torch.exp(log_probs - log_sum_all)


def compute_reconstruction_loss(y, recon_params):
    """
    Compute reconstruction loss based on output distribution
    """
    if len(recon_params.shape) == len(y.shape):  # Bernoulli case
        return torch.nn.functional.binary_cross_entropy(recon_params, y, reduction='sum')
    else:  # Gaussian case
        mu_recon, logvar_recon = recon_params
        recon_loss = -0.5 * (logvar_recon + (y - mu_recon).pow(2) * torch.exp(-logvar_recon))
        return -recon_loss.sum()  # Negative log-likelihood
    

def cond_prior(mu_post,
               var_post,
               mu_beta,
               var_beta,
               pi=None,
               M=10,
               ):
        
        
        
        K = mu_post.shape[0]
        ## TODO figure out what kind of distribution this is 
        dist_post = torch.distributions.normal.Normal(
            loc=mu_post,
            scale = torch.sqrt(var_post))
        
        # draw M samples from the normal distribution
        x = torch.normal(mean=mu_post.repeat(M),
                         std=torch.sqrt(var_post).repeat(M))

        # draw M the prior of w
        w = torch.normal(mean = torch.zeors(M),
                         std = torch.ones(M))
        

        if not pi:
            pi = torch.ones(K).pow(-1)

        def mc_kl(self,
               x,
               mu_x, # qφx from GMVAE 
               var_x,
               mu_beta, ## this is given w 
               var_beta,
               pi):
            
            """
            compute loss for an individual monte carlo sample 

            x = latent representation 
            K = number of clusters in the mixture (default 16)
            mu_prior_conditional
            var_prior_conditional
            pi = mixing probability (shape is K x 1)
        
            """
            # TODO confirm this creates K independent
            # normals given vectors of length K
            npdf= torch.distributions.normal.Normal(loc=mu_beta,
                                                        scale = torch.sqrt(var_beta))
                
            E_j = pi * npdf.log_prob(x).exp()
            denom = torch.sum(E_j)

            prob_z = E_j / denom

            # ----- compute KL divergence per cluster ---- #
            phix_kl = []
            for k in K:
                mu_beta_k = None
                var_beta_k = None
                kls += calc_kl(mu=mu_x,
                    logvar=logvar_x,
                    mu_o=mu_beta_k, 
                    var_o=var_beta_k,
                    reduce='none')
            phix_kl = torch.stack(phix_kl,dim=0)
            
            return torch.sum(prob_z*phix_kl) 

        return self.mc_mog_kl(z,mu_post,var_post,mu_beta,var_beta,pi)

