from abc import ABC, abstractmethod
import itertools
from typing import Any, Generic, Literal, ParamSpec, TypeVar, cast
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from dsh.utils.functions import create_cosine_similarity_cross_matrix, hamming_distance_via_cosine_similarity
from dsh.utils.stopwatch import GlobalProfiler
from dsh.utils.types import Quantize

Modality = Literal["image"] | Literal["text"]

_FP = ParamSpec("_FP")
_FR = TypeVar("_FR")


class LossBase(nn.modules.loss._Loss, ABC, Generic[_FP, _FR]):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    @abstractmethod
    def forward(self, *args: _FP.args, **kwargs: _FP.kwargs) -> _FR: ...

    def __call__(self, *args: _FP.args, **kwargs: _FP.kwargs) -> _FR:
        return cast(_FR, super().__call__(*args, **kwargs))


class PairwiseNLLLoss(LossBase[[Tensor, Tensor, Tensor], Tensor]):
    """Pairwise Negative Log Likelihood Loss with Similarity Matrix"""

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, similarity: Tensor, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the pairwise loss between an input tensor and a target tensor using negative log likelihood.

        Args:
            similarity (Tensor): A tensor of shape (n, n) where each element represents the similarity between two elements in the pred and target tensors.
            pred (Tensor): The predicted output tensor of shape (n, k).
            target (Tensor): The true labels or values tensor of shape (n, k).
        Returns:
            Tensor: A scalar loss value.
        """
        gamma = 0.5 * torch.matmul(pred, target.T)
        loss = -torch.sum(similarity * gamma - torch.log(1 + torch.exp(gamma)))

        return loss


class QuantizationLoss(LossBase[[Tensor, Tensor, Modality], Tensor], ABC):
    """Quantization Loss base class."""

    def __init__(
        self,
        quantization: Quantize,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.quantization = quantization

    @abstractmethod
    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        pass


class CrossModalityQuantizationLoss(QuantizationLoss):
    """Quantization Loss with Squared Frobenius Norm and with a target quantization tensor."""

    def __init__(
        self,
        quantization: Quantize,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(quantization, size_average, reduce, reduction)

    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        """Compute the Quantization Loss using Squared Frobenius Norm."""
        q_target = self.quantization(target)
        q_error = q_target - pred
        squared_frobenius_norm = torch.sum(torch.square(q_error))
        return squared_frobenius_norm


class AlwaysOneModalityQuantizationLoss(CrossModalityQuantizationLoss):
    """Quantization Loss with Squared Frobenius Norm where the target quantization tensor is always from the same modality no matter what pred is given."""

    def __init__(self, quantization: Quantize, modality: Modality, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(quantization, size_average, reduce, reduction)
        self.modality = modality

    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        if pred_modality == self.modality:
            return super().forward(pred, pred, pred_modality)
        else:
            return super().forward(pred, target, pred_modality)


class SameModalityQuantizationLoss(CrossModalityQuantizationLoss):
    """Quantization Loss with Squared Frobenius Norm where the target quantization tensor is from the same modality as the predicted one."""

    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        return super().forward(pred, pred, pred_modality)


class AverageHashCodeQuantizationLoss(QuantizationLoss):
    """Quantization Loss with Squared Frobenius Norm"""

    def __init__(
        self,
        quantization: Quantize,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(quantization, size_average, reduce, reduction)

    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        """Compute the Quantization Loss using Squared Frobenius Norm."""
        q_pred = self.quantization(pred)
        q_target = self.quantization(target)
        q_error = (q_pred + q_target) / 2.0 - pred
        squared_frobenius_norm = torch.sum(torch.square(q_error))
        return squared_frobenius_norm


class AverageActivationToHashCodeQuantizationLoss(QuantizationLoss):
    """Quantization Loss with Squared Frobenius Norm where the target tensor is obtained from all prediction modalities."""

    def __init__(
        self,
        quantization: Quantize,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(quantization, size_average, reduce, reduction)

    def forward(self, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        """Compute the Quantization Loss using Squared Frobenius Norm."""
        # TDSRDH paper states: B = sign(ɑ (X + Y)), however ɑ is irrelevant due to sign function
        q_target = self.quantization(pred + target)
        q_error = q_target - pred
        squared_frobenius_norm = torch.sum(torch.square(q_error))
        return squared_frobenius_norm


class DefaultQuantizationLoss(LossBase[[Tensor, Tensor], Tensor]):
    """Quantization Loss with Frobenius or L1 Norm where target tensor is obtained from all prediction modalities."""

    def __init__(
        self,
        quantization: Quantize,
        use_frobenius: bool,
        normalize_by_batch_size: bool,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.quantization = quantization
        self.use_frobenius = use_frobenius
        self.normalize_by_batch_size = normalize_by_batch_size

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute the Quantization Loss using Frobenius or L1 Norm."""
        q_target = self.quantization(pred + target)
        q_error = q_target - pred
        loss: Tensor
        if self.use_frobenius:
            loss = torch.norm(q_error)
        else:
            loss = q_error.abs().sum()
        if self.normalize_by_batch_size:
            n, _ = pred.shape
            loss /= n
        return loss


class CrossModalityTripletLoss(LossBase[[Tensor, Tensor], Tensor]):
    """Triplet Loss with Margin where positive and negative points stem from different modality than anchor points."""

    def __init__(self, epsilon: float = 1.0, vectorized: bool = True, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self.epsilon = epsilon

        self.forward = self.forward_vectorized if vectorized else self.forward_looped

    def forward(self, *args, **kwargs) -> Any:
        raise RuntimeError("This method gets selected based on the vectorized flag")

    def forward_looped(self, anchor: Tensor, other: Tensor) -> Tensor:
        """Compute the Triplet Loss with Margin.
        Args:
            anchor (Tensor): A tensor of shape (n, k) representing the anchor points.
            other (Tensor): A tensor of shape (n, k) representing the positive or negative points.
        Returns:
            Tensor: The computed Triplet Loss with Margin.
        """
        # Positive points are same indices, negative points are different indices
        n, _ = anchor.shape
        total_loss = torch.tensor(0.0).to(anchor.device)
        for i in range(n):
            X_i = anchor[i]
            Y_i = other[i]
            for j in range(n):
                if i != j:
                    Y_j = other[j]
                    distance_pos = (X_i - Y_i).pow(2).sum()
                    distance_neg = (X_i - Y_j).pow(2).sum()
                    loss = F.relu(self.epsilon + distance_pos - distance_neg)
                    total_loss += loss
        return total_loss

    def forward_vectorized(self, anchor: Tensor, other: Tensor) -> Tensor:
        """Compute the Triplet Loss with Margin in a vectorized way.
        Args:
            anchor (Tensor): A tensor of shape (n, k) representing the anchor points.
            other (Tensor): A tensor of shape (n, k) representing the positive or negative points.
        Returns:
            Tensor: The computed Triplet Loss with Margin.
        """
        n, _ = anchor.shape
        distance_pos = torch.square(anchor - other).unsqueeze(0).expand((n, n, -1)).sum(dim=2)
        distance_neg = torch.square(anchor.unsqueeze(0).expand((n, n, -1)) - other.unsqueeze(1).expand((n, n, -1))).sum(dim=2)
        loss = self.epsilon + distance_pos - distance_neg
        mask = torch.eye(n, device=anchor.device, dtype=torch.bool)
        loss[mask] = 0  # Set the diagonal elements to 0 since we don't want self-comparisons
        loss = F.relu(loss)
        total_loss = torch.sum(loss)
        return total_loss


class TDSRDHPaperLoss(LossBase[[Tensor, Tensor, Tensor, Modality], Tensor]):
    """Wrapper for Pairwise, Quantization and Triplet Losses."""

    def __init__(
        self,
        pairwise_loss: PairwiseNLLLoss,
        quantization_loss: QuantizationLoss,
        triplet_loss: CrossModalityTripletLoss,
        alpha: float,
        beta: float,
        normalize_losses_by_batch_size: bool,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.pairwise_loss = pairwise_loss
        self.quantization_loss = quantization_loss
        self.triplet_loss = triplet_loss
        self.alpha = alpha
        self.beta = beta
        self.normalize_losses_by_batch_size = normalize_losses_by_batch_size

    def forward(self, similarity: Tensor, pred: Tensor, target: Tensor, pred_modality: Modality) -> Tensor:
        with GlobalProfiler().step("TDSRDHPaperLoss") as s:
            p = self.pairwise_loss(similarity, pred, target)
            s.record("Pairwise")
            q = self.quantization_loss(pred, target, pred_modality)
            s.record("Quantization")
            t = self.triplet_loss(pred, target)
            s.record("Triplet")

            if self.normalize_losses_by_batch_size:
                batch_size = pred.shape[0]
                p = p / batch_size**2
                q = q / batch_size
                t = t / batch_size**2

            total_loss = p + self.alpha * q + self.beta * t
            return total_loss


class SCHPaperLoss(LossBase[[Tensor, tuple[Tensor, ...]], Tensor]):
    """Loss function as described in "Cross-Modal Hashing Method With Properties of Hamming Space: A New Perspective"."""

    def __init__(
        self,
        tau: int,
        alpha: float,
        beta: float,
        lneg_lambda_l: float,
        normalize_losses_by_batch_size: bool = True,
        use_k_approximation: bool = False,
        vectorized: bool | None = True,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        assert tau > 0, f"Parameter tau must be a positive integer, got {tau}"
        super().__init__(size_average, reduce, reduction)
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.lneg_lambda_l = lneg_lambda_l
        self.normalize_losses_by_batch_size = normalize_losses_by_batch_size
        self.use_k_approximation = use_k_approximation

        if vectorized != None:
            self.forward = self._forward_vectorized if vectorized else self._forward_iterative

    def forward(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("SCHPaperLoss-Both"):
            i = self._forward_iterative(similarity, preds)
            v = self._forward_vectorized(similarity, preds)

            assert torch.allclose(i, v)
            return v

    def _assert_inputs(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> tuple[int, int, torch.device]:
        with GlobalProfiler().step("Assert Inputs"):
            assert len(preds) > 0, f"Expected at least one input tensor, got {len(preds)}."
            n, k = preds[0].shape
            device = preds[0].device
            assert (
                similarity.shape[0] == n and similarity.shape[1] == n
            ), f"Similarity tensor must be of shape (n, n). Got {similarity.shape}."
            assert all(
                p.shape[0] == n and p.shape[1] == k for p in preds
            ), f"Prediction tensors must share same shape. Got: {[p.shape for p in preds]}."
            assert all(
                p.device == device for p in preds
            ), f"Prediction tensors must be on same device. Got: {[p.device for p in preds]}."
            return n, k, device

    def _forward_iterative(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("SCHPaperLoss-Iterative"):
            n, k, device = self._assert_inputs(similarity, preds)
            similarity = similarity.to(device)

            tensor_pairs: list[tuple[Tensor, Tensor]] = []
            if len(preds) == 1:
                (X,) = preds
                tensor_pairs.append((X, X))
            elif len(preds) == 2:
                X, Y = preds
                tensor_pairs.append((X, X))
                tensor_pairs.append((X, Y))
                tensor_pairs.append((Y, X))
                tensor_pairs.append((Y, Y))
            else:
                raise NotImplementedError()

            lneg_lambda_l = self.lneg_lambda_l * k

            total_loss = torch.zeros([], device=device)
            for p1, p2 in tensor_pairs:
                lnegs: list[Tensor] = []
                lpps: list[Tensor] = []
                lfps: list[Tensor] = []
                for i in range(n):
                    for j in range(n):
                        h = hamming_distance_via_cosine_similarity(p1[i], p2[j], self.use_k_approximation)
                        if torch.isclose(similarity[i][j], torch.zeros((1,), device=similarity.device)):
                            lnegs.append(lneg_lambda_l - h)
                        elif torch.isclose(similarity[i][j], torch.ones((1,), device=similarity.device)):
                            lfps.append(h - (k / 2 * (1 - similarity[i][j])))
                        else:
                            lambda_u = k / 2 * (1 - similarity[i][j])
                            lambda_l = lambda_u - self.tau
                            lpps.append(lambda_l - h)
                            lpps.append(h - lambda_u)
                lpos = torch.clamp(torch.cat([torch.tensor(lpps), torch.tensor(lfps) * self.alpha]), min=0).norm()
                lneg = torch.clamp(torch.tensor(lnegs) * self.beta, min=0).norm()
                total_loss += lpos + lneg

            if self.normalize_losses_by_batch_size:
                total_loss /= n * n
            return total_loss

    def _forward_vectorized(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("SCHPaperLoss-Vectorized"):
            n, k, device = self._assert_inputs(similarity, preds)
            similarity = similarity.to(device)

            mask_sim_zero = similarity.isclose(torch.zeros_like(similarity))
            mask_sim_one = similarity.isclose(torch.ones_like(similarity))

            lambda_u = k / 2 * (-similarity + 1.0)  # (1 - similarity) is reversed to preserve type information
            lambda_l = lambda_u - self.tau
            lambda_l[lambda_l < 0] = 0
            lambda_l[mask_sim_zero] = self.lneg_lambda_l * k

            # referenced code: https://github.com/hutt94/SCH/blob/main/train.py#L101-L106
            W_l = torch.ones((n, n), device=device)
            W_l[mask_sim_zero] = self.beta  # paper states 0, but code states beta
            W_l[mask_sim_one] = 0  # paper states beta, but code states 0

            W_u = torch.ones((n, n), device=device)
            W_u[mask_sim_zero] = 0  # paper states alpha, but code states 0
            W_u[mask_sim_one] = self.alpha  # paper states 0, but code states alpha

            total_loss = torch.zeros([], device=device)
            for p in itertools.product(preds, preds):
                # preserve type information by reverse (1 - cosine_similarity)
                BBT = k / 2 * (-create_cosine_similarity_cross_matrix(p, None, self.use_k_approximation) + 1.0)
                l1 = W_l * F.relu(lambda_l - BBT)
                l2 = W_u * F.relu(BBT - lambda_u)
                loss = l1.norm() + l2.norm()  # take frobenius norm for each loss part and add together
                assert isinstance(loss, Tensor)  # type annotations
                total_loss += loss

            if self.normalize_losses_by_batch_size:
                total_loss /= n * n
            return total_loss


class ModSCHPaperLoss(SCHPaperLoss):
    """Loss function modified from "Cross-Modal Hashing Method With Properties of Hamming Space: A New Perspective"."""

    def __init__(
        self,
        tau: int,
        alpha: float,
        beta: float,
        lneg_lambda_l: float,
        normalize_losses_by_batch_size: bool = True,
        use_frobenius: bool = True,
        use_k_approximation: bool = False,
        vectorized: bool | None = True,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            tau,
            alpha,
            beta,
            lneg_lambda_l,
            normalize_losses_by_batch_size,
            use_k_approximation,
            vectorized,
            size_average,
            reduce,
            reduction,
        )
        self.use_frobenius = use_frobenius

    def _forward_vectorized(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("ModSCHPaperLoss-Vectorized"):
            n, k, device = self._assert_inputs(similarity, preds)
            similarity = similarity.to(device)

            mask_sim_zero = similarity.isclose(torch.zeros_like(similarity))
            mask_sim_one = similarity.isclose(torch.ones_like(similarity))

            lambda_u = k / 2 * (-similarity + 1.0)  # (1 - similarity) is reversed to preserve type information
            lambda_l = lambda_u - self.tau
            lambda_u[mask_sim_zero] = k
            lambda_l[lambda_l < 0] = 0
            lambda_l[mask_sim_zero] = self.lneg_lambda_l * k

            W = torch.ones((n, n), device=device)
            W[mask_sim_zero] = self.beta
            W[mask_sim_one] = self.alpha

            total_loss = torch.zeros([], device=device)
            for p in itertools.product(preds, preds):
                # preserve type information by reverse (1 - cosine_similarity)
                BBT = k / 2 * (-create_cosine_similarity_cross_matrix(p, None, self.use_k_approximation) + 1.0)
                l1 = W * F.relu(lambda_l - BBT)
                l2 = W * F.relu(BBT - lambda_u)
                if self.use_frobenius:
                    # take frobenius norm for each loss part and add together
                    loss = l1.norm() + l2.norm()
                    assert isinstance(loss, Tensor)  # type annotations
                else:
                    loss = l1.abs().sum() + l2.abs().sum()
                total_loss += loss

            if self.normalize_losses_by_batch_size:
                total_loss /= n * n
            return total_loss


class DSCHLoss(SCHPaperLoss):
    """Dynamic Semantic Channel Hashing Loss"""

    def __init__(
        self,
        tau: float,
        alpha: float,
        beta: float,
        lambda_neg: float,
        gamma_w: float = 2.0,
        gamma_l: float = 1.0,
        anchor: float = 0.0,
        normalize_losses_by_batch_size: bool = True,
        use_frobenius: bool = False,
        use_k_approximation: bool = False,
        vectorized: bool | None = True,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        assert not tau < 0, f"Parameter tau must be a non-negative, got {tau}"
        super().__init__(
            1,  # ignore tau here
            alpha,
            beta,
            lambda_neg,
            normalize_losses_by_batch_size,
            use_k_approximation,
            vectorized,
            size_average,
            reduce,
            reduction,
        )
        self.tau = tau
        self.gamma_w = gamma_w
        self.gamma_l = gamma_l
        self.anchor = anchor
        self.use_frobenius = use_frobenius

    def _forward_iterative(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("DSCHLoss-Iterative"):
            n, k, device = self._assert_inputs(similarity, preds)
            similarity = similarity.to(device)

            tensor_pairs: list[tuple[Tensor, Tensor]] = []
            if len(preds) == 1:
                (X,) = preds
                tensor_pairs.append((X, X))
            elif len(preds) == 2:
                X, Y = preds
                tensor_pairs.append((X, X))
                tensor_pairs.append((X, Y))
                tensor_pairs.append((Y, X))
                tensor_pairs.append((Y, Y))
            else:
                raise NotImplementedError()

            lambda_neg = self.lneg_lambda_l * k

            total_loss = torch.zeros([], device=device)
            for b1, b2 in tensor_pairs:
                lnegs: list[Tensor] = []
                lpps: list[Tensor] = []
                lfps: list[Tensor] = []
                for i in range(n):
                    for j in range(n):
                        h = hamming_distance_via_cosine_similarity(b1[i], b2[j], self.use_k_approximation)
                        w = torch.pow(1 - similarity[i][j], self.gamma_w) * (k - lambda_neg - self.tau) + self.tau
                        p = lambda_neg * (1 - similarity[i][j]) + (self.anchor - 1.0) * similarity[i][j] * self.tau

                        bucket: list[Tensor]
                        if torch.isclose(similarity[i][j], torch.zeros((1,), device=similarity.device)):
                            bucket = lnegs
                        elif torch.isclose(similarity[i][j], torch.ones((1,), device=similarity.device)):
                            bucket = lfps
                        else:
                            bucket = lpps
                        bucket.append(p - h)
                        bucket.append(h - (p + w))
                lpos = torch.clamp(torch.cat([torch.tensor(lpps), torch.tensor(lfps) * self.alpha]), min=0)
                lneg = torch.clamp(torch.tensor(lnegs) * self.beta, min=0)
                if self.gamma_l != 1.0:
                    lpos = lpos.pow(self.gamma_l)
                    lneg = lneg.pow(self.gamma_l)
                if self.use_frobenius:
                    lpos = lpos.norm()
                    lneg = lneg.norm()
                else:
                    lpos = lpos.sum()
                    lneg = lneg.sum()
                total_loss += lpos + lneg

            if self.normalize_losses_by_batch_size:
                total_loss /= n * n
            return total_loss

    def _forward_vectorized(self, similarity: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        with GlobalProfiler().step("DSCHLoss-Vectorized"):
            n, k, device = self._assert_inputs(similarity, preds)
            similarity = similarity.to(device)

            mask_sim_zero = similarity.isclose(torch.zeros_like(similarity))
            mask_sim_one = similarity.isclose(torch.ones_like(similarity))

            W = torch.ones((n, n), device=device)
            W[mask_sim_zero] = self.beta
            W[mask_sim_one] = self.alpha

            lambda_neg = self.lneg_lambda_l * k
            CW = torch.pow(-similarity + 1.0, self.gamma_w) * (k - lambda_neg - self.tau) + self.tau
            P = lambda_neg * (-similarity + 1.0) + (self.anchor - 1) * similarity * self.tau

            total_loss = torch.zeros([], device=device)
            for b in itertools.product(preds, preds):
                # preserve type information by reverse (1 - cosine_similarity)
                BBT = k / 2 * (-create_cosine_similarity_cross_matrix(b, None, self.use_k_approximation) + 1.0)
                l1 = W * F.relu(P - BBT)
                l2 = W * F.relu(BBT - (P + CW))
                if self.gamma_l != 1.0:
                    l1 = l1.pow(self.gamma_l)
                    l2 = l2.pow(self.gamma_l)
                if self.use_frobenius:
                    # take frobenius norm for each loss part and add together
                    loss = l1.norm() + l2.norm()
                    assert isinstance(loss, Tensor)  # type annotations
                else:
                    loss = l1.abs().sum() + l2.abs().sum()
                total_loss += loss

            if self.normalize_losses_by_batch_size:
                total_loss /= n * n
            return total_loss


class OurLoss(LossBase[[Tensor, tuple[Tensor, ...]], Tensor]):
    """Our custom loss function"""

    def __init__(
        self,
        semanticchannel: SCHPaperLoss,
        quantization: DefaultQuantizationLoss,
        q_weight: float | None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)

        self.semanticchannel = semanticchannel
        self.quantization = quantization
        self.q_weight = q_weight

    def forward(self, sim: Tensor, preds: tuple[Tensor, ...]) -> Tensor:
        assert len(preds) == 2, "Not implemented for more than two modalities"
        s = self.semanticchannel(sim, preds)
        if self.q_weight is not None and self.q_weight > 0:
            q1 = self.quantization(preds[0], preds[1])
            q2 = self.quantization(preds[1], preds[0])
            return s + self.q_weight * (q1 + q2)
        else:
            return s
