import torch
import pytest

from basisbench.models.taylor import Taylor

def test_taylor_degree():
    with pytest.raises(ValueError):
        Taylor(-1)
        
def test_number_of_coeffients():
    model = Taylor(6)
    
    assert model.coefficients.shape == (7,)
    assert sum(p.numel() for p in model.parameters()) == 7
    
def test_design_matrix():
    model = Taylor(degree=3)
    
    x = torch.tensor([0.0,1.0,2.0])
    design_matrix = model._design_matrix(x)
    expected_matrix = torch.tensor([
        [1.0,0.0,0.0,0.0],
        [1.0,1.0,1.0,1.0],
        [1.0,2.0,4.0,8.0]
    ])
    
    assert torch.allclose(
        design_matrix,
        expected_matrix
    )
    
def test_forward():
    model = Taylor(degree=2)
    
    with torch.no_grad():
        model.coefficients.copy_(
            torch.tensor([1.0,2.0,3.0])
        )
        
    x = torch.tensor([0.0,1.0,2.0])
    predictions = model(x)
    expected = torch.tensor([1.0,6.0,17.0])
    
    assert torch.allclose(
        predictions,
        expected
    )
    
def test_predictions_shape():
    model = Taylor(degree=5)
    
    x = torch.randn(32)
    predictions = model(x)
    
    assert predictions.shape == (32,)
    
def test_gradient_flow():
    model = Taylor(degree=3)
    
    x = torch.randn(5)
    y = torch.randn(5)
    
    predictions = model(x)
    loss = torch.mean((predictions - y) ** 2)
    loss.backward()
    
    assert model.coefficients.grad is not None
    assert model.coefficients.grad.shape == (4,)
    assert torch.isfinite(model.coefficients.grad).all()
    
def test_input_must_be_one_dimensional():
    model = Taylor(degree=5)
    
    x = torch.randn(32,2)
    
    with pytest.raises(ValueError):
        model(x)