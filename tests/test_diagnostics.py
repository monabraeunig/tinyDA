import pytest

import numpy as np
from scipy.stats import multivariate_normal

from tinyDA.sampler import sample
from tinyDA.posterior import Posterior
from tinyDA.proposal import GaussianRandomWalk
from tinyDA.distributions import DefaultGaussianLogLike
from tinyDA.link import Link

from tinyDA.diagnostics import (
    get_samples,
    to_xarray,
    to_inference_data,
)

import arviz as az
import xarray as xr

#--------------------------------------------------------------------------------------------
np.random.seed(42)

#--------------------------------------------------------------------------------------------
# Forward models
def forward_fine(theta):
    return np.array(theta)

def forward_medium(theta):
    return 0.95 * np.array(theta)

def forward_coarse(theta):
    return 0.90 * np.array(theta)

#--------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def prior():
    return multivariate_normal(mean=np.zeros(2), cov=np.eye(2))

@pytest.fixture(scope="session")
def data():
    np.random.seed(42)
    theta_true = np.array([0.1, -0.2])
    return forward_fine(theta_true) + 0.05 * np.random.randn(2)

@pytest.fixture(scope="session")
def posteriors(prior, data):
    lik_fine = DefaultGaussianLogLike(data, covariance=0.05 * np.eye(2))
    lik_medium = DefaultGaussianLogLike(data, covariance=0.1 * np.eye(2))
    lik_coarse = DefaultGaussianLogLike(data, covariance=0.2 * np.eye(2))

    return {
        "fine": Posterior(prior, lik_fine, model=forward_fine),
        "medium": Posterior(prior, lik_medium, model=forward_medium),
        "coarse": Posterior(prior, lik_coarse, model=forward_coarse),
    }

@pytest.fixture(scope="session")
def proposal():
    return GaussianRandomWalk(C=np.eye(2)*0.05, adaptive=False)

#--------------------------------------------------------------------------------------------
iterations = 500
n_chains = 1
subchain_length = 2
subchain_lengths = [2,2]
# MH
@pytest.fixture(scope="session")
def mh_chain(posteriors, proposal):
    return sample(
        posteriors=posteriors["fine"],
        proposal=proposal,
        iterations=iterations,
        n_chains=n_chains,
        force_sequential=True,
    )

# DA
@pytest.fixture(scope="session")
def da_chain(posteriors, proposal):
    return sample(
        posteriors=[posteriors["coarse"], posteriors["fine"]],
        proposal=proposal,
        iterations=iterations,
        n_chains=n_chains,
        subchain_length=subchain_length,
        randomize_subchain_length=False,
        store_coarse_chain=True,
        force_sequential=True,
    )

# MLDA (3 levels)
@pytest.fixture(scope="session")
def mlda_chain(posteriors, proposal):
    return sample(
        posteriors=[posteriors["coarse"], posteriors["medium"], posteriors["fine"]],
        proposal=proposal,
        iterations=iterations,
        n_chains=n_chains,
        subchain_length=subchain_lengths,
        randomize_subchain_length=False,
        store_coarse_chain=True,
        force_sequential=True,
    )

@pytest.fixture(scope="session")
def samples_mh(mh_chain): 
    return get_samples(mh_chain, attribute="parameters")

@pytest.fixture(scope="session")
def samples_da_coarse(da_chain):
    return get_samples(da_chain, attribute="parameters", level="fine")

@pytest.fixture(scope="session")
def samples_mlda_level0(mlda_chain): 
    return get_samples(mlda_chain, attribute="parameters", level=2)
#--------------------------------------------------------------------------------------------
# test get_samples()

@pytest.mark.parametrize(
    "chain_name, attribute, level, burnin",
    [
        # MH
        ("mh_chain", "parameters", "fine", 0),
        ("mh_chain", "model_output", "fine", 0),
        # qoi hat die chain dim 1 und alle einträge sind none
        ("mh_chain", "qoi", "fine", 0),  
        ("mh_chain", "stats", "fine", 0),

        #DA
        ("da_chain", "parameters", "fine", 0),
        ("da_chain", "model_output", "fine", 0),
        ("da_chain", "qoi", "fine", 0),
        ("da_chain", "stats", "fine", 0),

        ("da_chain", "parameters", "coarse", 0),
        ("da_chain", "model_output", "coarse", 0),
        ("da_chain", "qoi", "coarse", 0),
        ("da_chain", "stats", "coarse", 0),

        #MLDA
        ("mlda_chain", "parameters", 2, 0),
        ("mlda_chain", "model_output", 2, 0),
        ("mlda_chain", "qoi", 2, 0),
        ("mlda_chain", "stats", 2, 0),

        ("mlda_chain", "parameters", 1, 0),
        ("mlda_chain", "model_output", 1, 0),
        ("mlda_chain", "qoi", 1, 0),
        ("mlda_chain", "stats", 1, 0),

        ("mlda_chain", "parameters", 0, 0),
        ("mlda_chain", "model_output", 0, 0),
        ("mlda_chain", "qoi", 0, 0),
        ("mlda_chain", "stats", 0, 0),
    ],
)
def test_get_samples(request, chain_name, attribute, level, burnin):
    
    chain = request.getfixturevalue(chain_name)
    chain_samples = get_samples(chain=chain, attribute=attribute, level=level, burnin=burnin)
    # Test output 
    assert isinstance(chain_samples, dict)
    assert isinstance(chain_samples['chain_0'], np.ndarray)
    
    if (chain_name == "mh_chain"):
        assert chain_samples["sampler"] == "MH"
        # does get_samples get the correct parameters
        if (attribute == "parameters"):
            raw = np.array([link.parameters for link in chain["chain_0"]])
            assert np.allclose(chain_samples["chain_0"], raw)
            
    elif (chain_name == "da_chain"):
        assert chain_samples["sampler"] == "DA"
        assert chain_samples["subchain_length"] == subchain_length
        # does get_samples get the correct parameters
        if (attribute == "parameters"):
            if level == "fine":
                raw_chain = chain["chain_fine_0"]
            elif level == "coarse":
                raw_chain = chain["chain_coarse_0"]
                
            raw = np.array([link.parameters for link in raw_chain])
            assert np.allclose(chain_samples["chain_0"], raw)
            
    elif(chain_name == "mlda_chain"):
        assert chain_samples["sampler"] == "MLDA"
        assert chain_samples["subchain_lengths"] == subchain_lengths
        # does get_samples get the correct parameters
        if (attribute == "parameters"):
            if (level == 0):
                raw_chain = chain["chain_l0_0"]
            elif (level == 1):
                raw_chain = chain["chain_l1_0"]
            elif (level == 2):
                raw_chain = chain["chain_l2_0"]
    
            raw = np.array([link.parameters for link in raw_chain])
            assert np.allclose(chain_samples["chain_0"], raw)
        
    assert chain_samples["n_chains"] == n_chains

    if (level == "fine" or level == 2):
        assert chain_samples["iterations"] == iterations + 1
    elif (level == "coarse" or level == 1):
        assert chain_samples["iterations"] == iterations * 2
    elif (level == 0):
        assert chain_samples["iterations"] == iterations * 4
    
    if (attribute == "parameters" or attribute == "model_output"):
        assert chain_samples["dimension"] == 2
    elif (attribute == "qoi"):
        assert chain_samples["dimension"] == 1
    elif (attribute == "stats"):
        assert chain_samples["dimension"] == 3




#--------------------------------------------------------------------------------------------
# test to_xarray()

@pytest.mark.parametrize(
    "samples_name, keys",
    [
        ("samples_mh", ["x0","x1"]),
        ("samples_da_coarse", ["x0","x1"]),
        ("samples_mlda_level0", ["x0","x1"]),
    ],
)
def test_to_xarray(request, samples_name, keys):
    #MH samples
    samples = request.getfixturevalue(samples_name)
    xarr = to_xarray(samples, keys=keys)

    assert isinstance(xarr, xr.Dataset)
    assert "x0" in xarr
    assert "x1" in xarr

    original = samples["chain_0"]

    # test value concistency 
    assert np.allclose(xarr["x0"].values[0], original[:, 0])
    assert np.allclose(xarr["x1"].values[0], original[:, 1])
    
    if (samples.get("level") in (None, "fine", 2)):
        assert xarr["x0"].shape == (1, iterations + 1)
        assert xarr["x1"].shape == (1, iterations + 1)
    elif (samples["level"] == "corase" or samples["level"] == 1):
        assert xarr["x0"].shape == (1, iterations * 2)
        assert xarr["x1"].shape == (1, iterations * 2)
    elif (samples["level"] == 0):
        assert xarr["x0"].shape == (1, iterations * 4)
        assert xarr["x1"].shape == (1, iterations * 4)

#--------------------------------------------------------------------------------------------
# Test to_inference_data()

@pytest.mark.parametrize(
    "chain_name, level, burnin, parameter_names",
    [
        ("mh_chain", "fine", 0, None),
        ("da_chain", "fine", 0, None),
        ("mlda_chain", 2, 0, None),
        
    ],
)
def test_to_inference_data(request, chain_name, level, burnin, parameter_names):
    chain = request.getfixturevalue(chain_name)
    idata = to_inference_data(chain, level, burnin, parameter_names)
    
    assert isinstance(idata, az.InferenceData)
    for group in ["posterior", "posterior_predictive", "sample_stats", "qoi"]:
        assert group in idata.groups()

    #test data variables
    assert len(idata.posterior.data_vars) > 0
    for var in idata.posterior.data_vars:
        values = idata.posterior[var].values
        assert values.size > 0
        
    assert len(idata.posterior_predictive.data_vars) > 0
    for var in idata.posterior_predictive.data_vars:
        values = idata.posterior_predictive[var].values
        assert values.size > 0

    for key in ["prior", "likelihood", "posterior"]:
        assert key in idata.sample_stats.data_vars

    # test values consistency
    samples = get_samples(chain, "parameters", level=level)
    assert np.allclose(idata.posterior["x0"].values[0], samples["chain_0"][:, 0])

    # ich weiß nicht genau was ich mit qoi machen soll weil das ist immer none

#--------------------------------------------------------------------------------------------
# Test get_twolevel_inference_data
    # gibts nicht


            
#--------------------------------------------------------------------------------------------
# Test get_promoted_samples
    # gibts nicht


#--------------------------------------------------------------------------------------------
# test get_multilevel_inference_data
    # gibts nicht
