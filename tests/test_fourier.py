import pytest
import torch

from basisbench.models.fourier import Fourier


def test_fourier_initializes():
    """Fourier model should initialize with a valid frequency."""
    model = Fourier(max_frequency=5)

    assert model.max_frequency == 5

import pytest
import torch

from basisbench.models.fourier import Fourier


def test_fourier_initializes():
    """A valid maximum frequency should initialize the model."""
    model = Fourier(max_frequency=5)

    assert model.max_frequency == 5


@pytest.mark.parametrize("frequency", [0, -1, -5])
def test_fourier_rejects_non_positive_frequency(frequency):
    """Maximum frequency must be greater than zero."""
    with pytest.raises(ValueError):
        Fourier(max_frequency=frequency)


@pytest.mark.parametrize("frequency", [1.5, 2.0, "5", None])
def test_fourier_rejects_non_integer_frequency(frequency):
    """Maximum frequency must be an integer."""
    with pytest.raises(ValueError):
        Fourier(max_frequency=frequency)
        
@pytest.mark.parametrize("frequency", [1, 4, 5, 10])
def test_fourier_output_shape(frequency):
    """Output should have the same shape as the input."""
    model = Fourier(max_frequency=frequency)

    X = torch.linspace(-1, 1, 20)

    output = model(X)

    assert output.shape == X.shape
    
def test_fourier_output_shape_2d():
    """Fourier should preserve a 2-D input shape."""
    model = Fourier(max_frequency=5)

    X = torch.linspace(-1, 1, 20).reshape(-1, 1)

    output = model(X)

    assert output.shape == X.shape
    
def test_fourier_output_is_finite():
    """Fourier basis should not produce NaN or infinite values."""
    model = Fourier(max_frequency=10)

    x = torch.linspace(-10, 10, 100)

    y = model(x)

    assert torch.isfinite(y).all()


@pytest.mark.parametrize('frequency', [1,4,5,10])
def test_fourier_parameter_count(frequency):
    """
    For K frequencies:

        y = b + sum_k [a_k cos(kx) + b_k sin(kx)]

    There should be:

        1 + 2K

    trainable parameters.
    """
    model = Fourier(max_frequency=frequency)
    
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    assert parameter_count == 1 + 2 * frequency


def test_fourier_has_gradients():
    """Model parameters should receive gradients during backpropagation."""
    model = Fourier(max_frequency=5)

    x = torch.linspace(-1, 1, 20)
    target = torch.sin(x)

    prediction = model(x)
    loss = ((prediction - target) ** 2).mean()

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_fourier_can_overfit_simple_sine():
    """
    A Fourier model containing frequency 3 should be able to
    represent sin(3x).
    """
    torch.manual_seed(42)

    model = Fourier(max_frequency=3)

    x = torch.linspace(-1, 1, 200)
    target = torch.sin(3 * x)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.05,
    )

    for _ in range(500):
        optimizer.zero_grad()

        prediction = model(x)
        loss = ((prediction - target) ** 2).mean()

        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-2
    
    
def test_fourier_forward_matches_expected_function():
    """Verify the Fourier basis mathematically."""
    model = Fourier(max_frequency=2)

    with torch.no_grad():
        model.bias.fill_(2.0)

        model.cos_coefficients.zero_()
        model.sin_coefficients.zero_()

        # f(x) = 2 + 3*cos(x) - 4*sin(2x)
        model.cos_coefficients[0] = 3.0
        model.sin_coefficients[1] = -4.0

    x = torch.tensor([0.0, 0.5, 1.0])

    expected = (
        2.0
        + 3.0 * torch.cos(x)
        - 4.0 * torch.sin(2 * x)
    )

    actual = model(x)

    assert torch.allclose(actual, expected, atol=1e-3)
    
def test_fourier_uses_max_frequency():
    """
    The model must actually use the highest requested frequency.
    """
    model = Fourier(max_frequency=5)

    with torch.no_grad():
        model.bias.zero_()
        model.cos_coefficients.zero_()
        model.sin_coefficients.zero_()

        # Only frequency 5 is active.
        model.cos_coefficients[4] = 1.0

    X = torch.tensor([0.0, 0.2, 0.7, 1.0])

    expected = torch.cos(5 * X)

    actual = model(X)

    assert torch.allclose(actual, expected)