import torch
import pytest

from basisbench.models.neural_network import NeuralNetwork

@pytest.mark.parametrize('hidden_size', [1,2,5,10])
def test_neural_network_initialization(hidden_size):
    nn = NeuralNetwork(hidden_size=hidden_size)
    assert nn.hidden_size == hidden_size
    
@pytest.mark.parametrize('hidden_size', [-1,-10])
def test_neural_network_rejects_non_positive_integers(hidden_size):
    with pytest.raises(ValueError):
        NeuralNetwork(hidden_size)
        
@pytest.mark.parametrize('hidden_size', ['10', None])
def test_neural_network_rejects_non_integer_hidden_size(hidden_size):
    with pytest.raises(ValueError):
        NeuralNetwork(hidden_size)
        
@pytest.mark.parametrize('hidden_size', [1,2,3,4,5])
def test_the_number_of_parameters(hidden_size):
    nn = NeuralNetwork(hidden_size)
    
    parameter_count = sum(
        parameter.numel()
        for parameter in nn.parameters()
    )
    
    assert parameter_count ==  3 * hidden_size + 1
    
@pytest.mark.parametrize('hidden_size', [2,5,10])
def test_output_shape(hidden_size):
    nn = NeuralNetwork(hidden_size)
    x = torch.linspace(-1,1,100)
    
    output = nn(x)
    
    assert output.shape == x.shape
    
@pytest.mark.parametrize('batch_size', [1,2,3])
def test_neural_network_only_accepts_1D_inputs(batch_size):
    "Network expects a 1d input"
    nn = NeuralNetwork(hidden_size=10)
    x = torch.randn(batch_size,batch_size,1)
    
    with pytest.raises(ValueError):
        nn(x)
        
def test_neural_network_forward_matches_formula():
    """Verify the forward pass against the mathematical definition."""
    model = NeuralNetwork(hidden_size=2)

    with torch.no_grad():
        model.w1.copy_(
            torch.tensor([2.0, -1.0])
        )

        model.b1.copy_(
            torch.tensor([0.5, -0.5])
        )

        model.w2.copy_(
            torch.tensor([3.0, 4.0])
        )

        model.b2.fill_(1.0)

    x = torch.tensor([0.0, 0.5, 1.0])

    expected = (
        3.0 * torch.tanh(2.0 * x + 0.5)
        + 4.0 * torch.tanh(-x - 0.5)
        + 1.0
    )

    actual = model(x)

    assert torch.allclose(actual, expected)
        
def test_neural_network_zero_weights_returns_bias():
    """With all weights and hidden biases zero, output equals b2."""
    model = NeuralNetwork(hidden_size=10)

    with torch.no_grad():
        model.w1.zero_()
        model.b1.zero_()
        model.w2.zero_()
        model.b2.fill_(5.0)

    x = torch.linspace(-10, 10, 100)

    output = model(x)

    expected = torch.full_like(x, 5.0)

    assert torch.allclose(output, expected)
    
def test_neural_network_backward():
    """All parameters should receive finite gradients."""
    model = NeuralNetwork(hidden_size=10)

    x = torch.linspace(-1, 1, 100)
    target = x**3

    prediction = model(x)

    loss = ((prediction - target) ** 2).mean()

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        
def test_neural_network_output_is_finite():
    """Forward pass should not produce NaN or infinity."""
    model = NeuralNetwork(hidden_size=20)

    x = torch.linspace(-100, 100, 1_000)

    output = model(x)

    assert torch.isfinite(output).all()
    
def test_neural_network_can_learn_simple_function():
    """Network should be able to approximate a simple nonlinear function."""
    torch.manual_seed(42)

    model = NeuralNetwork(hidden_size=10)

    x = torch.linspace(-1, 1, 200)
    target = x**2

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
    )

    for _ in range(1000):
        optimizer.zero_grad()

        prediction = model(x)

        loss = ((prediction - target) ** 2).mean()

        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1