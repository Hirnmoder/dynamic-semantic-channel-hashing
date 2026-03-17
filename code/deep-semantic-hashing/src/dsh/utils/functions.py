import numpy as np
import torch

__all__ = [
    "sign_zero_is_positive",
    "sign_zero_is_negative",
    "sign_zero_is_zero",
    "create_similarity_cross_matrix",
    "create_cosine_similarity_cross_matrix",
    "hamming_distance",
    "hamming_distance_via_cosine_similarity",
]


def sign_zero_is_positive(input: torch.Tensor) -> torch.Tensor:
    """The sign function that returns 1 for positive numbers and zero, and returns -1 for negative numbers."""
    return torch.where(input >= 0, 1, -1)


def sign_zero_is_negative(input: torch.Tensor) -> torch.Tensor:
    """The sign function that returns 1 for positive numbers, and returns -1 for negative numbers and zero."""
    return torch.where(input > 0, 1, -1)


def sign_zero_is_zero(input: torch.Tensor) -> torch.Tensor:
    """The sign function that returns 1 for positive numbers, 0 for zero, and -1 for negative numbers."""
    return torch.sign(input)


def create_similarity_cross_matrix(labels: torch.Tensor) -> torch.Tensor:
    """Create a similarity matrix from the input tensor.
    Args:
       labels (Tensor): A binary tensor of shape (n, l) where n is the number of samples and l is the number of labels.
    Returns:
       Tensor: A similarity matrix of shape (n, n) where each element is 0 if no labels match and 1 if at least one label matches. The diagonal is set to 1.
    """
    eye = torch.eye(labels.shape[0], device=labels.device).bool()
    similarity_matrix = (eye | (labels.unsqueeze(1) & labels.unsqueeze(0)).any(dim=2)).float()
    return similarity_matrix


def create_cosine_similarity_cross_matrix(
    X: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    correlate_diagonal: float | None = None,
    use_k_approximation: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create a cosine-similarity cross-matrix from the input tensor(s).
    `create_cosine_similarity_cross_matrix((X, X), ...) == create_cosine_similarity_cross_matrix(X, ...)`
    Args:
        X (Tensor | tuple[Tensor, Tensor]): Either one or two tensor(s) of shape (n, k) or where n is the number of samples and k is the dimensionality of the vectors to compute the cosine-similarity on.
        correlate_diagonal (float | None): The value to set on the diagonal of the cosine similarity matrix, usually 1. If None, no specific value is set.
        use_k_approximation (bool): If True, use sqrt(k) as an approximation of the length of each vector.
        eps (float): A small value to avoid division by zero.
    Returns:
        Tensor: A cosine-similarity matrix of shape (n, n) where each element is the cosine similarity between the rows of X.
    """
    if isinstance(X, tuple):
        X1, X2 = X
    else:
        X1, X2 = X, X
    assert len(X1.shape) == 2, f"Expected 2 dimensional input tensor, got {X1.shape}."
    assert len(X2.shape) == 2, f"Expected 2 dimensional input tensor, got {X2.shape}."
    n, k = X1.shape
    assert n == X2.shape[0] and k == X2.shape[1], f"Shape mismatch between input tensors, got {X1.shape} != {X2.shape}."
    if use_k_approximation:
        cosine_similarity = (X1.float() @ X2.float().T) / k
    else:
        cosine_similarity = torch.cosine_similarity(X1.float().unsqueeze(1), X2.float().unsqueeze(0), dim=2, eps=eps)
    if correlate_diagonal != None:
        cosine_similarity.fill_diagonal_(correlate_diagonal)
    return cosine_similarity


def hamming_distance(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Calculate the Hamming distance between two binary vectors.
    Args:
        x (np.ndarray): A numpy array of shape (k,) representing a binary vector.
        Y (np.ndarray): A numpy array of shape (m, k) representing another binary vector.
    Returns:
        np.ndarray: The Hamming distance between the two vectors (m,)
    """
    assert x.shape[0] == Y.shape[1]
    assert x.dtype == np.bool and Y.dtype == np.bool, f"Expected bool dtype for inputs, got {x.dtype}, {Y.dtype}"
    return np.sum(x ^ Y, axis=1)


def hamming_distance_via_cosine_similarity(x: torch.Tensor, y: torch.Tensor, use_k_approximation: bool) -> torch.Tensor:
    """Calculate the Hamming distance between two real-valued vectors with elements in [-1, 1].
    Args:
        x (torch.Tensor): A tensor of shape (k,) representing a first real-valued vector.
        y (torch.Tensor): A tensor of shape (k,) representing a second real-valued vector.
        use_k_approximation (bool): If True, use sqrt(k) as an approximation of the length of each vector.
    Returns:
        torch.Tensor: The Hamming distance between the two vectors (1,)
    """
    assert x.shape == y.shape
    (k,) = x.shape
    if use_k_approximation:
        cosine_similarity = (x * y).sum() / k
    else:
        cosine_similarity = (x * y).sum() / (x.norm(2) * y.norm(2))
    return k / 2 * (1 - cosine_similarity)
