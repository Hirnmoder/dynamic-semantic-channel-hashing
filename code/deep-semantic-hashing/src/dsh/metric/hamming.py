from dataclasses import dataclass
from typing import Iterable
import h5py
import multiprocessing as mp
import numpy as np

import dsh.config.env
from dsh.config.metric import HammingMetricConfiguration, AveragePrecisionMode

from dsh.metric.metricbase import MetricBase

from dsh.utils.functions import hamming_distance
from dsh.utils.logger import Logger
from dsh.utils.metrics import roc_auc
from dsh.utils.progress import tqdm
from dsh.utils.stopwatch import StopWatch
from dsh.utils.tensorboard import Writer
from dsh.utils.types import CM, Constants, MetricEvalSet, T


@dataclass
class HammingRankingResult:
    average_precision_img2text: float
    average_precision_text2img: float
    average_precision_img2img: float
    average_precision_text2text: float

    @staticmethod
    def ZERO() -> "HammingRankingResult":
        return HammingRankingResult(0.0, 0.0, 0.0, 0.0)


@dataclass
class HashLookupResult:
    img2text: dict[int, CM]
    text2img: dict[int, CM]
    img2img: dict[int, CM]
    text2text: dict[int, CM]

    @staticmethod
    def ZERO(thresholds: Iterable[int], tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0) -> "HashLookupResult":
        d = {threshold: CM(tp, fp, tn, fn) for threshold in thresholds}
        return HashLookupResult(d, d, d, d)


@dataclass
class HammingResult:
    query_index: int
    hamming_ranking: HammingRankingResult
    hash_lookup: HashLookupResult
    message: str | None = None


@dataclass
class HammingRankingResults:
    mean_average_precision_img2text: float
    mean_average_precision_text2img: float
    mean_average_precision_img2img: float
    mean_average_precision_text2text: float
    average_precisions_img2text: dict[int, float]
    average_precisions_text2img: dict[int, float]
    average_precisions_img2img: dict[int, float]
    average_precisions_text2text: dict[int, float]


@dataclass
class HashLookupResults:
    roc_auc_img2text: float
    roc_auc_text2img: float
    roc_auc_img2img: float
    roc_auc_text2text: float
    roc_curve_img2text: dict[int, tuple[float, float]]
    roc_curve_text2img: dict[int, tuple[float, float]]
    roc_curve_img2img: dict[int, tuple[float, float]]
    roc_curve_text2text: dict[int, tuple[float, float]]
    cumulated_img2text: dict[int, CM]
    cumulated_text2img: dict[int, CM]
    cumulated_img2img: dict[int, CM]
    cumulated_text2text: dict[int, CM]


@dataclass
class HammingResults:
    hamming_ranking: HammingRankingResults
    hash_lookup: HashLookupResults


GlobalCalcContextType = tuple[bool, AveragePrecisionMode, np.ndarray, np.ndarray, str]
LocalCalcContextType = tuple[int]


class HammingMetric(MetricBase[HammingMetricConfiguration, h5py.File, HammingResults]):
    def __init__(self, config: HammingMetricConfiguration, env: dsh.config.env.MetricEnvironmentConfig):
        super().__init__(config, env)

    @property
    def name(self) -> str:
        return Constants.Metrics.HammingMetric

    def calculate(self, data: h5py.File, writer: Writer) -> HammingResults:
        sw = StopWatch(None, f"{self.namewithset} calculate")
        assert Constants.H5.Labels in data, "Labels not found in HDF5 file"
        assert Constants.H5.Image.BoolHash in data, "Image hash not found in HDF5 file"
        assert Constants.H5.Text.BoolHash in data, "Text hash not found in HDF5 file"
        assert Constants.H5.Sets.SubsetMembership in data, "Subset information not found in HDF5 file"

        labels = data[Constants.H5.Labels]
        assert isinstance(labels, h5py.Dataset), "Labels must be a dataset"
        img_hash = data[Constants.H5.Image.BoolHash]
        assert isinstance(img_hash, h5py.Dataset), "Image hash must be a dataset"
        text_hash = data[Constants.H5.Text.BoolHash]
        assert isinstance(text_hash, h5py.Dataset), "Text hash must be a dataset"
        assert img_hash.shape == text_hash.shape, "Image hash and text hash must have the same shape"
        hash_len = img_hash.shape[1]

        sets = data[Constants.H5.Sets.SubsetMembership]
        assert isinstance(sets, h5py.Dataset), "Subset information must be a dataset"
        match self.config.eval_set:
            case MetricEvalSet.TEST:
                query_column_index = Constants.H5.Sets.ColumnIndex.Test
                retrieval_column_index = Constants.H5.Sets.ColumnIndex.TestRetrieval
            case MetricEvalSet.VAL:
                query_column_index = Constants.H5.Sets.ColumnIndex.Validation
                retrieval_column_index = Constants.H5.Sets.ColumnIndex.ValidationRetrieval
            case _:
                raise NotImplemented(f"Unknown metric evaluation set {self.config.eval_set}")
        query_indices = np.argwhere(sets[:, query_column_index]).squeeze()
        retrieval_indices = np.argwhere(sets[:, retrieval_column_index]).squeeze()

        # H5 file includes top-k labels specified during training only
        # if specified, subselect top k labels for metric calculation
        if self.config.subselect_top_k_labels != None:
            assert (
                self.config.subselect_top_k_labels <= labels.shape[1]
            ), "subselect_top_k_labels must be less than or equal to the number of labels"
            relevant_label_indices = np.argsort(np.sum(labels, axis=0))[::-1][: self.config.subselect_top_k_labels]
        else:
            relevant_label_indices = np.arange(labels.shape[1])

        labels = np.array(labels)[:, relevant_label_indices]

        avg_precisions_img2text: dict[int, float] = {}
        avg_precisions_text2img: dict[int, float] = {}
        avg_precisions_img2img: dict[int, float] = {}
        avg_precisions_text2text: dict[int, float] = {}

        hash_lookup_img2text: dict[int, dict[int, CM]] = {}
        hash_lookup_text2img: dict[int, dict[int, CM]] = {}
        hash_lookup_img2img: dict[int, dict[int, CM]] = {}
        hash_lookup_text2text: dict[int, dict[int, CM]] = {}

        # loop through all query samples, parallelized
        _global_calc_context: GlobalCalcContextType = (
            self.config.include_samples_without_labels,
            self.config.average_precision_mode,
            labels,
            retrieval_indices,
            data.filename,
        )
        with mp.Pool(
            min(mp.cpu_count(), self.env.num_workers),
            initializer=_initialize_worker_process,
            initargs=[_global_calc_context],
        ) as pool:
            _local_calc_contexts: list[LocalCalcContextType] = []
            for query_index in query_indices:
                _local_calc_contexts.append((int(query_index),))
            i = pool.imap_unordered(_calculate_metrics_of_sample, _local_calc_contexts, self.env.samples_per_worker)
            for result in tqdm(i, total=len(_local_calc_contexts)):
                if result == None:
                    continue
                if isinstance(result, str):
                    Logger().info(result)
                else:
                    if result.message != None:
                        Logger().info(result.message)
                    # Hamming Ranking results
                    avg_precisions_img2text[result.query_index] = result.hamming_ranking.average_precision_img2text
                    avg_precisions_text2img[result.query_index] = result.hamming_ranking.average_precision_text2img
                    avg_precisions_img2img[result.query_index] = result.hamming_ranking.average_precision_img2img
                    avg_precisions_text2text[result.query_index] = result.hamming_ranking.average_precision_text2text
                    # Hash Lookup results
                    hash_lookup_img2text[result.query_index] = result.hash_lookup.img2text
                    hash_lookup_text2img[result.query_index] = result.hash_lookup.text2img
                    hash_lookup_img2img[result.query_index] = result.hash_lookup.img2img
                    hash_lookup_text2text[result.query_index] = result.hash_lookup.text2text

        # Calculate Mean Average Precisions (MAP)
        map_img2text = float(np.mean(list(avg_precisions_img2text.values()))) if len(avg_precisions_img2text) > 0 else 0.0
        map_text2img = float(np.mean(list(avg_precisions_text2img.values()))) if len(avg_precisions_text2img) > 0 else 0.0
        map_img2img = float(np.mean(list(avg_precisions_img2img.values()))) if len(avg_precisions_img2img) > 0 else 0.0
        map_text2text = float(np.mean(list(avg_precisions_text2text.values()))) if len(avg_precisions_text2text) > 0 else 0.0

        writer.add_scalar(T.metric(self.set, T.HAMMING_RANKING, T.IMG2TEXT), map_img2text, 0.5)
        writer.add_scalar(T.metric(self.set, T.HAMMING_RANKING, T.TEXT2IMG), map_text2img, 0.5)
        writer.add_scalar(T.metric(self.set, T.HAMMING_RANKING, T.IMG2IMG), map_img2img, 0.5)
        writer.add_scalar(T.metric(self.set, T.HAMMING_RANKING, T.TEXT2TEXT), map_text2text, 0.5)

        # Calculate Cumulated Confusion Matrices
        ccm_img2text = {t: CM.cumulate([hl[t] for hl in hash_lookup_img2text.values()]) for t in range(hash_len + 1)}
        ccm_text2img = {t: CM.cumulate([hl[t] for hl in hash_lookup_text2img.values()]) for t in range(hash_len + 1)}
        ccm_img2img = {t: CM.cumulate([hl[t] for hl in hash_lookup_img2img.values()]) for t in range(hash_len + 1)}
        ccm_text2text = {t: CM.cumulate([hl[t] for hl in hash_lookup_text2text.values()]) for t in range(hash_len + 1)}
        # Calculate ROC Curves and AUCs
        roc_curve_img2text = {t: (cm.false_positive_rate, cm.true_positive_rate) for t, cm in ccm_img2text.items()}
        roc_curve_text2img = {t: (cm.false_positive_rate, cm.true_positive_rate) for t, cm in ccm_text2img.items()}
        roc_curve_img2img = {t: (cm.false_positive_rate, cm.true_positive_rate) for t, cm in ccm_img2img.items()}
        roc_curve_text2text = {t: (cm.false_positive_rate, cm.true_positive_rate) for t, cm in ccm_text2text.items()}
        roc_auc_img2text = roc_auc(roc_curve_img2text.values())
        roc_auc_text2img = roc_auc(roc_curve_text2img.values())
        roc_auc_img2img = roc_auc(roc_curve_img2img.values())
        roc_auc_text2text = roc_auc(roc_curve_text2text.values())

        writer.add_pr_curve_raw(T.metric(self.set, T.HASH_LOOKUP, T.PR, T.IMG2TEXT), ccm_img2text, hash_len + 1, 0.5)
        writer.add_pr_curve_raw(T.metric(self.set, T.HASH_LOOKUP, T.PR, T.TEXT2IMG), ccm_text2img, hash_len + 1, 0.5)
        writer.add_pr_curve_raw(T.metric(self.set, T.HASH_LOOKUP, T.PR, T.IMG2IMG), ccm_img2img, hash_len + 1, 0.5)
        writer.add_pr_curve_raw(T.metric(self.set, T.HASH_LOOKUP, T.PR, T.TEXT2TEXT), ccm_text2text, hash_len + 1, 0.5)

        writer.add_scalar(T.metric(self.set, T.HASH_LOOKUP, T.ROC_AUC, T.IMG2TEXT), roc_auc_img2text, 0.5)
        writer.add_scalar(T.metric(self.set, T.HASH_LOOKUP, T.ROC_AUC, T.TEXT2IMG), roc_auc_text2img, 0.5)
        writer.add_scalar(T.metric(self.set, T.HASH_LOOKUP, T.ROC_AUC, T.IMG2IMG), roc_auc_img2img, 0.5)
        writer.add_scalar(T.metric(self.set, T.HASH_LOOKUP, T.ROC_AUC, T.TEXT2TEXT), roc_auc_text2text, 0.5)

        # Finally, track some time
        writer.add_scalar(T.times(self.set, T.HAMMING), sw.record(f"Finish calculating {self.namewithset}").total_duration, 0.5)

        results = HammingResults(
            HammingRankingResults(
                map_img2img,
                map_img2text,
                map_text2img,
                map_text2text,
                avg_precisions_img2text,
                avg_precisions_text2img,
                avg_precisions_img2img,
                avg_precisions_text2text,
            ),
            HashLookupResults(
                roc_auc_img2text,
                roc_auc_text2img,
                roc_auc_img2img,
                roc_auc_text2text,
                roc_curve_img2text,
                roc_curve_text2img,
                roc_curve_img2img,
                roc_curve_text2text,
                ccm_img2text,
                ccm_text2img,
                ccm_img2img,
                ccm_text2text,
            ),
        )
        return results


GLOBAL_CALC_CONTEXT: tuple[*GlobalCalcContextType, np.ndarray, np.ndarray, int]|None = None # fmt: skip
def _initialize_worker_process(_calc_context: GlobalCalcContextType):
    global GLOBAL_CALC_CONTEXT
    include_samples_without_labels, average_precision_mode, labels, retrieval_indices, h5_filename = _calc_context

    with h5py.File(h5_filename, "r") as h5_file:
        img_hash = h5_file[Constants.H5.Image.BoolHash]
        assert isinstance(img_hash, h5py.Dataset), "Image hash must be a dataset"
        text_hash = h5_file[Constants.H5.Text.BoolHash]
        assert isinstance(text_hash, h5py.Dataset), "Text hash must be a dataset"
        ih = np.array(img_hash)
        th = np.array(text_hash)

        GLOBAL_CALC_CONTEXT = (
            include_samples_without_labels,
            average_precision_mode,
            labels,
            retrieval_indices,
            h5_filename,
            ih,
            th,
            ih.shape[1],
        )


def _calculate_metrics_of_sample(_calc_context: LocalCalcContextType) -> HammingResult | str | None:
    global GLOBAL_CALC_CONTEXT
    (query_index,) = _calc_context

    if GLOBAL_CALC_CONTEXT is None:
        raise SystemError("GLOBAL_CALC_CONTEXT is not set")
    include_samples_without_labels, average_precision_mode, labels, retrieval_indices, _, ih, th, hash_len = GLOBAL_CALC_CONTEXT

    q_ih: np.ndarray = ih[query_index]
    q_th: np.ndarray = th[query_index]
    q_lb: np.ndarray = labels[query_index]

    if not q_lb.any():
        if include_samples_without_labels:
            return HammingResult(query_index, HammingRankingResult.ZERO(), HashLookupResult.ZERO(range(hash_len + 1)))
        else:
            return  # skip this query sample as it has no labels

    r_ih = ih[retrieval_indices]
    r_th = th[retrieval_indices]
    r_lb = labels[retrieval_indices]

    n = retrieval_indices.shape[0]
    related_samples = (q_lb & r_lb).any(axis=1).flatten()  # (n,) bool array
    k = related_samples.sum(dtype=np.int32)
    if k == 0:
        msg = f"[MET] No related samples found for query sample {query_index}"
        if include_samples_without_labels:
            return HammingResult(query_index, HammingRankingResult.ZERO(), HashLookupResult.ZERO(range(hash_len + 1)), msg)
        else:
            return msg  # skip this query sample as it has no labels

    # Calculate Hamming distances between query sample and retrieval samples
    hamming_distances_img2text = hamming_distance(q_ih, r_th)
    hamming_distances_text2img = hamming_distance(q_th, r_ih)
    hamming_distances_img2img = hamming_distance(q_ih, r_ih)
    hamming_distances_text2text = hamming_distance(q_th, r_th)

    # Calculate Average Precision (AP) for each ranking result
    hrr: HammingRankingResult
    match average_precision_mode:
        case AveragePrecisionMode.LIKE_TDSRDH_PAPER | AveragePrecisionMode.DEFAULT:
            # Sort by Hamming distance
            indices_img2text = np.argsort(hamming_distances_img2text)
            indices_text2img = np.argsort(hamming_distances_text2img)
            indices_img2img = np.argsort(hamming_distances_img2img)
            indices_text2text = np.argsort(hamming_distances_text2text)

            # Hamming Ranking
            hr_img2text = np.cumsum(related_samples[indices_img2text], dtype=np.int32)
            hr_text2img = np.cumsum(related_samples[indices_text2img], dtype=np.int32)
            hr_img2img = np.cumsum(related_samples[indices_img2img], dtype=np.int32)
            hr_text2text = np.cumsum(related_samples[indices_text2text], dtype=np.int32)

            match average_precision_mode:
                case AveragePrecisionMode.LIKE_TDSRDH_PAPER:
                    hrr = HammingRankingResult(
                        float(np.sum(hr_img2text[:k] / np.arange(1, k + 1)) / k),
                        float(np.sum(hr_text2img[:k] / np.arange(1, k + 1)) / k),
                        float(np.sum(hr_img2img[:k] / np.arange(1, k + 1)) / k),
                        float(np.sum(hr_text2text[:k] / np.arange(1, k + 1)) / k),
                    )
                case AveragePrecisionMode.DEFAULT:
                    hrr = HammingRankingResult(
                        float(np.sum(hr_img2text * related_samples[indices_img2text] / np.arange(1, n + 1)) / k),
                        float(np.sum(hr_text2img * related_samples[indices_text2img] / np.arange(1, n + 1)) / k),
                        float(np.sum(hr_img2img * related_samples[indices_img2img] / np.arange(1, n + 1)) / k),
                        float(np.sum(hr_text2text * related_samples[indices_text2text] / np.arange(1, n + 1)) / k),
                    )
                case _:
                    raise NotImplementedError("Internal implementation error")
        case AveragePrecisionMode.TIE_AWARE:
            hrr = HammingRankingResult(
                _calculate_tie_aware_average_precision_np(hamming_distances_img2text, related_samples, hash_len),
                _calculate_tie_aware_average_precision_np(hamming_distances_text2img, related_samples, hash_len),
                _calculate_tie_aware_average_precision_np(hamming_distances_img2img, related_samples, hash_len),
                _calculate_tie_aware_average_precision_np(hamming_distances_text2text, related_samples, hash_len),
            )
        case _:
            raise ValueError(f"Unknown average precision mode {average_precision_mode}")

    # Hash lookup
    hl_img2text: dict[int, CM] = {}
    hl_text2img: dict[int, CM] = {}
    hl_img2img: dict[int, CM] = {}
    hl_text2text: dict[int, CM] = {}
    for distance in range(hash_len + 1):
        pos_img2text = hamming_distances_img2text <= distance
        hl_img2text[distance] = CM(
            int(np.sum(pos_img2text & related_samples)),
            int(np.sum(pos_img2text & ~related_samples)),
            int(np.sum(~pos_img2text & ~related_samples)),
            int(np.sum(~pos_img2text & related_samples)),
        )

        pos_text2img = hamming_distances_text2img <= distance
        hl_text2img[distance] = CM(
            int(np.sum(pos_text2img & related_samples)),
            int(np.sum(pos_text2img & ~related_samples)),
            int(np.sum(~pos_text2img & ~related_samples)),
            int(np.sum(~pos_text2img & related_samples)),
        )

        pos_img2img = hamming_distances_img2img <= distance
        hl_img2img[distance] = CM(
            int(np.sum(pos_img2img & related_samples)),
            int(np.sum(pos_img2img & ~related_samples)),
            int(np.sum(~pos_img2img & ~related_samples)),
            int(np.sum(~pos_img2img & related_samples)),
        )

        pos_text2text = hamming_distances_text2text <= distance
        hl_text2text[distance] = CM(
            int(np.sum(pos_text2text & related_samples)),
            int(np.sum(pos_text2text & ~related_samples)),
            int(np.sum(~pos_text2text & ~related_samples)),
            int(np.sum(~pos_text2text & related_samples)),
        )

    hlr = HashLookupResult(
        hl_img2text,
        hl_text2img,
        hl_img2img,
        hl_text2text,
    )

    return HammingResult(query_index, hrr, hlr)


def _calculate_tie_aware_average_precision_cmp_implementations(
    distances: np.ndarray, true_related: np.ndarray, hash_len: int
) -> float:
    result_it = _calculate_tie_aware_average_precision_it(distances, true_related, hash_len)
    result_np = _calculate_tie_aware_average_precision_np(distances, true_related, hash_len)
    if not np.isclose(result_it, result_np):
        print(f"Iterative and numpy-ed results differ: {result_it} vs {result_np}")
    return result_it


def _calculate_tie_aware_average_precision_it(distances: np.ndarray, true_related: np.ndarray, hash_len: int) -> float:
    # Calculation based on "Hashing as Tie-Aware Learning to Rank" by He, Cakir, Bargal, Scarloff (2018)
    ap_R: list[float] = []
    N_plus = np.sum(true_related)
    if N_plus == 0:
        return 0.0  # No positive samples, return 0 as AP
    n_d = [np.sum(distances == d) for d in range(hash_len + 1)]
    n_d_plus = [np.sum((distances == d) & true_related) for d in range(hash_len + 1)]
    N_d = np.cumsum(n_d)
    N_d_plus = np.cumsum(n_d_plus)
    for d in range(hash_len + 1):
        if n_d[d] == 0:
            # No samples with distance d, skip this step
            continue
        elif n_d[d] == 1:
            # Single sample with distance d means no permutations to consider
            ap_R.append(n_d_plus[d] * N_d_plus[d] / N_d[d])
        else:
            # Calculate average precision for distance d over all permutations
            c = 0.0
            lower_N_d = N_d[d - 1] if d > 0 else 0
            lower_N_d_plus = N_d_plus[d - 1] if d > 0 else 0
            for t in range(n_d[d]):
                c += (1 + lower_N_d_plus + t * ((n_d_plus[d] - 1) / (n_d[d] - 1))) / (1 + t + lower_N_d)
            ap_R.append(n_d_plus[d] * c / n_d[d])
    return float(np.sum(ap_R) / N_plus)


def _calculate_tie_aware_average_precision_np(distances: np.ndarray, true_related: np.ndarray, hash_len: int) -> float:
    # Calculation based on "Hashing as Tie-Aware Learning to Rank" by He, Cakir, Bargal, Scarloff (2018)
    # Implementation based on the paper's code: https://github.com/kunhe/TALR/blob/master/%2Beval/tieAP.m
    N_plus = np.sum(true_related)
    if N_plus == 0:
        return 0.0  # No positive samples, return 0 as AP
    n_d = np.array([np.sum(distances == d) for d in range(hash_len + 1)])
    n_d_plus = np.array([np.sum((distances == d) & true_related) for d in range(hash_len + 1)])
    N_d = np.cumsum(n_d)
    N_d_plus = np.cumsum(n_d_plus)

    ap_R = np.zeros(hash_len + 1)
    # Avoid division by zero for ranks with no positive samples
    mask = N_d > 0
    ap_R[mask] = n_d_plus[mask] * N_d_plus[mask] / N_d[mask]

    for d in np.argwhere((n_d > 1) & (n_d_plus > 0)).flatten().tolist():
        N_d_plus_lower = N_d_plus[d - 1] if d > 0 else 0
        N_d_lower = N_d[d - 1] if d > 0 else 0
        sum_numerator = N_d_plus_lower + 1 + np.arange(n_d[d]) * (n_d_plus[d] - 1) / (n_d[d] - 1)
        sum_denominator = np.arange(N_d_lower, N_d[d]) + 1
        ap_R[d] = n_d_plus[d] / n_d[d] * np.sum(sum_numerator / sum_denominator)

    return float(np.sum(ap_R) / N_plus)
