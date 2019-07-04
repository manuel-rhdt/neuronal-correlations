# Renormalisation Group Formalism

We know from statistical physics that systems close to the critical point show homogeneous scaling in lots of thermodynamic parameters. This behaviour is closely related to self-similarity of critical systems at various scales. To analyse the critical dynamics of systems the renormalization group formalism has been very successful.

We will review the basics of renormalisation using the Ising model as an explanatory device. We will also show you how renormalisation can be used as a data analysis tool to show critical behaviour in non-physical systems.

## Real Space Renormalisation

Let's say we have a system described by the dynamical variables $\{ \sigma_i \}$. In the Ising model these would be the individual spins $\sigma_i = \pm 1$. A renormalisation step entails going from a system described by $\{ \sigma_i \}$ to a system described by $\{ \tilde\sigma_{\tilde\imath} \}$ where the number of variables $\tilde\sigma_{\tilde\imath}$ after renormalisation is smaller than the number of original variables $\sigma_i$. This renormalisation step is commonly called *coarse-graining*.

![Coarse graining for a 2D ising lattice.](Figure.png)

At the beginning we describe our system by a joint probability distribution function (or equivalently by an effective Hamiltonian) over the original variables. A coarse-graining step then leads to a flow in the space of models

$$
P(\{\sigma_i\}) \longrightarrow \tilde P(\{\tilde\sigma_i\})
$$

called RG-flow.

The change of variables from $\{ \sigma_i \}$ to $\{ \tilde\sigma_{\tilde\imath} \}$ has to be chosen based on the details of the model to be analyzed. (You may remember from ASP that the decimation procedure for the 1D Ising model did not work for higher dimensional Ising lattices). 

- For the Ising model it makes sense to cluster together the nearest neighbour spins at each renormalization step, since these influence each other most strongly (interactions are *local*).
- For neurons spatial closeness is not related to the interaction strength between sites. Therefore it makes no sense to cluster spatially close neurons together.

## Free Energy

From the analysis of the Ising model we know that at the critical point the (nonanalytical part of the) free energy obeys the scaling law

$$
f_s(\{\sigma_i\}) = b^{-d} f_s (\{\tilde\sigma_i\})
$$

where $b$ is the decimation factor and $d$ is a characteristic scaling exponent ($b=2$ and $d=1$ for a simple decimation procedure in 1D Ising).

We want to find a similar scaling law for the activity of neurons which leads to the question of which quantity of the biological system is analogous to the free energy. If we model the neurons as spins $\sigma_i\in\{0,1\}$ with a joint probability density

$$
P(\{\sigma_i\}) = \frac1Z\exp\left[ \sum\limits_i h_i \sigma_i + \sum\limits_{i≠j} J_{ij}\sigma_i\sigma_j + \cdots \right]
$$

we see that the probability of complete silence in the cluster

$$
P(\{ \sigma_i = 0 \}) = \frac1Z = e^F
$$

is a direct function of the free energy. Therefore we expect $\ln P(\{ \sigma_i = 0 \})$ to show power-law behaviour as a function of cluster size. If $K$ is the cluster size we thus expect a scaling relationship

$$
P(\{ \sigma_i = 0 \}; K) = \exp\left(-a K^\beta\right)
$$

where $\beta$ is an empirical scaling exponent.

*The data very clearly demonstrates that $\ln P(\{ \sigma_i = 0 \})$ scales as a power law.*

## Probability Distribution

We know from statistical mechanics that the RG-flow has a fixed point when the system is critical (in other words: a system at the critical point is scale invariant). Therefore we want to show that the RG-Flow of the neurons exhibits a non-trivial fixed point. How do we expect the joint probability function to change under an RG-step for a system close to criticality?

If the neurons were nearly completely uncorrelated the central limit theorem would drive the distribution of the coarse-grained variables toward a fixed gaussian form. If the RG-flow has a nontrivial fixed point then there exists a regime were the distribution will stay non-gaussian even after repeted coarse-graining.

*In the results of the paper we see a clear non-gaussian distribution. The probability distributions of different RG-steps fall on top of each other.*

# Topics of secondary importance

## Momentum Space Renormalization 

In real space RG we used an ad-hoc clustering scheme to reduce the number of degrees of freedom in every coarse-graining step. For the Ising model it makes sense to successively integrate out the short-wavelength modes of the system.

## Equilibrium Statistical Physics

In statistical physics the joint probability distribution of interest is given by

$$
P \left(\left\{ \sigma_i \right\}\right) = \frac1Z e^{-\beta H \left(\left\{ \sigma_i \right\}\right)} \;.
$$

The renormalization procedure is done such that the partition function $Z$ does not change with each renormalisation step. This transformation consists of two steps: *decimation* and *relabeling*. Let's consider the one-dimensional Ising model

$$
H = -J \sum\limits_{\langle ij \rangle} \sigma_i \sigma_j - h\sum\limits_i \sigma_i
$$

Looking at the partition function we can perform the partial trace over the odd spins

$$
Z = \sum\limits_{\{\sigma_i = \pm 1\}} e^{\beta J (\sigma_1\sigma_2 + \sigma_2\sigma_3 + \ldots)} = \sum\limits_{\{\sigma_i = \pm 1\}} \prod\limits_{i=2,4,6} e^{\beta J \sigma_i(\sigma_{i-1} + \sigma_{i+1})} = \sum\limits_{\sigma_{i\in \text{odd} = \pm 1}} \prod\limits_{i=2,4,6} \left[e^{\beta J(\sigma_{i-1} + \sigma_{i+1})} + e^{-\beta J (\sigma_{i-1} + \sigma_{i+1})}\right]
$$

and then relabel the indices

$$
Z = \sum\limits_{\{\tilde\sigma_i = \pm 1\}} \prod\limits_{i} \left[ e^{\beta J (\tilde\sigma_i + \tilde\sigma_{i+1})} + e^{-\beta J (\tilde\sigma_i + \tilde\sigma_{i+1})} \right] = 
$$
