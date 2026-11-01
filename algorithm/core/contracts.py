"""Dictionary contracts shared by the notebook and visual search functions."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray


BoundingBox = tuple[int, int, int, int]
"""(x_min, y_min, x_max, y_max) in pixels, with exclusive maxima."""

MatchAnswer = Literal["Yes", "No", "Uncertain"]


class MatchRating(TypedDict):
    """Validated visual assessment; score is an ordinal rating, not a probability."""

    score: float  # Match rating normalized to [0, 1]; display as 100 * score.
    answer: MatchAnswer  # Whether all requested attributes are visibly supported.
    reason: str  # Short explanation grounded in visible image evidence.


class VerifierScore(MatchRating):
    """A validated match rating with model identity."""

    model_id: str  # Actual verifier model; authoritative provenance.
    raw_response: NotRequired[
        str
    ]  # Exact successful local-model text before JSON normalization.
    prompt: NotRequired[
        str
    ]  # Actual local verifier prompt, including repair text if used.
    input_size: NotRequired[
        tuple[int, int]
    ]  # PIL input dimensions before processor resizing.
    model_revision: NotRequired[str]


class SlicClusterAssignments(TypedDict):
    """Minimal input from cluster_slic_regions; both arrays share the same order."""

    region_ids: NDArray[np.integer[Any]]  # Unique SLIC label IDs present in the image.
    region_clusters: NDArray[np.integer[Any]]  # Cluster ID assigned to each region ID.


class RankedCluster(TypedDict):
    """A whole masked region and its visual verifier's query-match assessment."""

    cluster_id: int  # K-means cluster identifier.
    region_ids: list[int]  # SLIC labels included in the union mask.
    n_regions: int  # Number of SLIC labels in the cluster.
    pixel_count: int  # Number of selected pixels before cropping.
    image: NDArray[np.uint8]  # Tight RGB crop, black outside the union mask.
    bbox: BoundingBox  # Cluster crop bounds in original-image coordinates.
    score: float  # Verifier match rating in [0, 1].
    verifier: VerifierScore  # Actual visual verifier assessment.
    scoring_method: Literal["qwen3vl_local_visual_query_match"]
    score_components: NotRequired[RegionScoreComponents]


class SAM3Detection(TypedDict):
    """Proposal relative to the detector crop."""

    bbox: BoundingBox  # Detection bounds in supplied-image coordinates.
    score: float  # Detector confidence before candidate ranking.
    sam3_score: float | None  # SAM3 confidence.
    mask: NDArray[np.bool_]  # Binary mask with the supplied image's spatial shape.
    image: NDArray[np.uint8]  # RGB box crop from the supplied image.
    prompt: str  # Text concept passed to SAM3.


class SAM3Region(SAM3Detection):
    """A cluster detection mapped to the original image; mask stays cluster-local."""

    cluster_id: int  # Parent K-means cluster identifier.
    cluster_score: float  # Parent cluster's verifier rating in [0, 1].
    local_bbox: BoundingBox  # Detection bounds in cluster-crop coordinates.
    # In this record bbox uses original-image coordinates, unlike SAM3Detection.


class EvidenceInfo(TypedDict):
    """Coordinates and rendered sizes of the actual context/detail scoring input."""

    source_bbox: BoundingBox  # Original SAM3 candidate; never enlarged geometrically.
    context_bbox: BoundingBox  # Wider, unmasked original-image window.
    detail_bbox: BoundingBox  # Candidate plus a small margin, in original coordinates.
    canvas_size: tuple[int, int]  # Uploaded evidence-sheet width and height.
    source_detail_size: tuple[int, int]  # Native detail pixels before enlargement.
    rendered_detail_size: tuple[int, int]  # Detail size actually drawn in the sheet.
    context_target_bbox: BoundingBox  # Marked target in canvas coordinates.
    detail_target_bbox: BoundingBox  # Marked target in canvas coordinates.


class ProposalEvidence(EvidenceInfo):
    """Lossless RGB input sent to the visual verifier and its coordinate mapping."""

    image: NDArray[np.uint8]


class CandidateCheck(TypedDict):
    """One SAM3 box verified on a context/detail sheet, or left unchecked."""

    rank: int  # One-based SAM3 confidence order within the region.
    detection_index: int  # Zero-based index in the NMS-filtered detection list.
    local_bbox: BoundingBox  # Box in region-crop coordinates.
    bbox: BoundingBox  # Box in original-image coordinates.
    sam3_score: float | None  # Detector confidence for this candidate.
    checked: bool  # Whether Qwen received this candidate's evidence sheet.
    candidate_score: float  # Qwen/SAM3 fused proposal score.
    qwen: NotRequired[VerifierScore]  # Per-candidate visual verdict when checked.
    evidence: NotRequired[ProposalEvidence]  # Actual context/detail sheet.


class CandidatePrompt(TypedDict):
    """Serializable metadata for one numbered SAM3 box on a whole-region sheet."""

    index: int  # One-based box number drawn on the image.
    label: str  # Detector concept shown beside the box.
    confidence: float | None  # SAM3 confidence, when available.
    bbox: BoundingBox  # Box in the supplied region image's coordinates.


class RegionAssessment(TypedDict):
    """Whole-region verifier output: ranking plus per-candidate ratings."""

    ranking: list[int]  # One-based candidate indices, best first.
    best_index: int  # Equal to ranking[0].
    ambiguous: bool  # Tied/weak/inconsistent top candidates need detail fallback.
    ratings: dict[int, MatchRating]  # Every candidate index exactly once.


class RankedRegion(SAM3Region):
    """A proposal ranked by per-candidate verification; score is its fused score.

    image is the raw detector box crop. When evidence is present, evidence.image
    is the actual context/detail verifier input.
    """

    verifier: VerifierScore  # Actual candidate assessment or a not-checked sentinel.
    scoring_method: Literal["qwen3vl_local_visual_query_match"]
    evidence: NotRequired[ProposalEvidence]  # Actual enlarged context/detail input.
    scoring_scope: NotRequired[
        Literal["region", "proposal"]
    ]  # Region scores do not verify each box.
    candidate_rank: NotRequired[int]  # One-based SAM3 confidence order.
    candidate_score: NotRequired[float]  # Explicit alias of score for proposal scope.
    candidate_verified: NotRequired[bool]  # Qwen said Yes above the acceptance score.
    candidate_qwen: NotRequired[VerifierScore]  # Verdict when this box was checked.
    score_components: NotRequired[RegionScoreComponents]


class SlicConfig(TypedDict):
    """Parameters forwarded to skimage.segmentation.slic."""

    n_segments: int  # Requested superpixel count; optimization overrides this field.
    compactness: float  # Spatial regularization relative to appearance similarity.
    sigma: float  # Gaussian smoothing before segmentation.
    max_num_iter: int  # Maximum SLIC refinement iterations.
    enforce_connectivity: bool  # Require connected superpixels.
    convert2lab: bool  # Convert the displayed PCA RGB to Lab before SLIC.
    slic_zero: bool  # Enable SLIC-zero's adaptive compactness.


class SlicMetrics(TypedDict):
    """Color-entropy statistics for one candidate SLIC segmentation."""

    n_segments: int
    actual_regions: int
    mean_entropy: float
    p90_entropy: float
    max_entropy: float


class SlicOptimization(TypedDict):
    """Winning segmentation and the metrics used to select it."""

    labels: NDArray[np.intp]
    config: SlicConfig
    metrics: SlicMetrics
    region_ids: NDArray[np.integer[Any]]
    region_entropies: NDArray[np.float64]
    history: list[SlicMetrics]


class SlicClusters(SlicClusterAssignments):
    """K-means grouping of superpixels in the displayed DINO PCA colors."""

    labels: NDArray[np.integer[Any]]  # Cluster ID at each local pixel.
    region_features: NDArray[np.float64]  # Mean three-channel color per superpixel.
    model: object  # Fitted sklearn KMeans instance; not serialized in reports.


class SearchConfig(TypedDict):
    """Explicit controls for the bounded recursive visual search."""

    strategy: Literal["agent-uct", "uct", "greedy"]  # Branch allocation rule.
    device: str  # auto, cpu, or a CUDA device for both local models.
    max_api_calls: int  # Hard HTTP-attempt budget; 0 means unlimited.
    max_iterations: int  # Maximum completed/attempted frontier expansions.
    max_depth: int  # Maximum crop depth; root is depth 0.
    branching_factor: int  # Requested semantic child groups per expansion.
    min_region_pixels: int  # Stop splitting regions with fewer active pixels.
    max_proposals_per_node: int  # SAM3 crops assessed per expansion; 0 means all.
    exploration_weight: float  # UCT exploration coefficient.
    cost_weight: float  # Penalty applied to normalized predicted remaining cost.
    reference_cost_seconds: float  # Cost normalization scale for branch selection.
    sam3_threshold: float  # Strict minimum detector confidence for proposals.
    mask_threshold: float  # SAM3 binary-mask threshold.
    sam_prompt: str  # Detector concept; an empty string uses the full query.
    rewrite_sam_prompt: NotRequired[
        bool
    ]  # Ask local Qwen for an object-first detector phrase.
    sam3_first: NotRequired[
        bool
    ]  # Detect first, then verify candidate boxes; False keeps the legacy path.
    verification_mode: NotRequired[
        Literal["region", "candidate"]
    ]  # Verify the whole cluster together or one evidence sheet per candidate.
    discovery_mode: NotRequired[
        Literal["off", "sam3_qwen"]
    ]  # Seed the tree from tiled SAM3 candidates reranked by Qwen.
    discovery_tiles: NotRequired[int]  # Grid side for the initial SAM3 scan.
    discovery_candidates: NotRequired[int]  # SAM3 candidates reranked by Qwen.
    discovery_seeds: NotRequired[int]  # Top reranked candidates added as tree seeds.
    discovery_overlap: NotRequired[float]  # Fractional tile overlap for discovery.
    region_candidate_limit: NotRequired[
        int
    ]  # Top SAM3 boxes drawn for whole-region verification; 0 disables.
    region_fallback_limit: NotRequired[
        int
    ]  # Ambiguous regions re-check this many candidates with detail sheets.
    candidate_check_limit: NotRequired[
        int
    ]  # Maximum candidate evidence sheets checked per region; 0 disables.
    candidate_accept_score: NotRequired[
        float
    ]  # Minimum Qwen score treated as a verified candidate.
    candidate_qwen_weight: NotRequired[float]  # Per-candidate Qwen score weight.
    candidate_sam3_weight: NotRequired[float]  # Per-candidate SAM3 confidence weight.
    candidate_query_mode: NotRequired[
        Literal["full", "target_phrase"]
    ]  # Query text shown with each candidate evidence sheet.
    feature_max_side: int  # Maximum side for each fresh DINO crop encoding.
    segmentation_max_side: int  # Bound SLIC work; labels map back to original pixels.
    verification_max_side: int  # Maximum side of the context/detail evidence sheet.
    context_padding: int  # Minimum context margin in original pixels per side.
    detail_padding: int  # Minimum margin around the target in the enlarged detail.
    n_segments_values: tuple[int, ...]  # SLIC candidates, selected by entropy.
    entropy_bins: int  # Quantization bins per PCA-color channel.
    slic: SlicConfig  # Shared SLIC settings copied before optimization.
    stop_on_match: bool  # Stop after a localized Yes above match_threshold.
    match_threshold: float  # Heuristic rating threshold, not a calibrated probability.
    max_local_scores: NotRequired[
        int
    ]  # Local generation budget including repair attempts; 0 is unlimited.
    cluster_spatial_weight: NotRequired[float]  # Normalized centroid weight in K-means.
    min_cluster_occupancy: NotRequired[float]  # Active-pixel fraction of a child bbox.
    max_cluster_aspect: NotRequired[float]  # Maximum child bbox elongation.
    min_largest_component: NotRequired[
        float
    ]  # Minimum dominant connected-component fraction.
    proposal_backend: NotRequired[
        str
    ]  # Recorded to prevent incompatible batch resumes.
    scoring_backend: NotRequired[str]  # Exact local model ID and revision.
    region_qwen_weight: NotRequired[
        float
    ]  # Relative weight of question-specific region relevance.
    region_confidence_weight: NotRequired[
        float
    ]  # Weight of mean confidence after duplicate suppression.
    region_count_weight: NotRequired[float]  # Weight of capped distinct-box count.
    region_diversity_weight: NotRequired[float]  # Weight of low pairwise box IoU.
    region_box_count_cap: NotRequired[
        int
    ]  # Number of distinct boxes at which the count reward saturates.
    region_nms_iou: NotRequired[
        float
    ]  # Suppress lower-confidence boxes above this IoU.


class RegionScoreComponents(TypedDict):
    """Auditable region reward; every component is bounded in [0, 1]."""

    qwen_score: (
        float  # Best candidate Qwen score, or whole-region score in legacy mode.
    )
    raw_box_count: int  # Above-threshold detections before NMS; may be detector-capped.
    unique_box_count: int  # Distinct boxes remaining after confidence-ordered NMS.
    mean_confidence: float  # Mean SAM3 confidence of verified boxes; zero if none.
    count_score: float  # min(verified_box_count, cap) / cap.
    mean_pairwise_iou: float  # Mean IoU among verified boxes; zero with fewer than two.
    diversity_score: float  # 1 - mean_pairwise_iou; zero if no verified boxes.
    combined_score: float  # Normalized weighted sum used as the tree reward.
    gate: NotRequired[
        Literal["no_candidates", "verified", "unverified", "budget"]
    ]  # Why the region reward is zero, when applicable.
    verified_box_count: NotRequired[int]  # Candidates Qwen accepted above threshold.
    candidate_checks: NotRequired[int]  # Evidence sheets sent to Qwen for this region.


class ChildRegion(TypedDict):
    """A child crop strictly contained in its parent's active pixels."""

    bbox: BoundingBox  # Original-image coordinates.
    mask: NDArray[np.bool_]  # Active pixels within bbox; black holes remain excluded.
    region_ids: NotRequired[list[int]]  # Local SLIC IDs for semantic partitions.


class SearchNode(ChildRegion):
    """Allocation statistics and spatial state of one immutable region."""

    node_id: int
    parent_id: int  # -1 for the root.
    depth: int
    children: list[int]
    visits: int  # Completed evaluations backed up through this subtree.
    reward_sum: float  # Sum of raw quality returns, without cost penalties.
    evaluated: bool  # Whether this node's complete evaluation finished.
    exhausted: bool  # No unevaluated descendants remain within the depth limit.
    assessment: NotRequired[VerifierScore]  # Cached best-candidate evidence.
    evaluation_error: NotRequired[
        str
    ]  # Recoverable verifier failure; no artificial reward is backed up.


class NodeEvidence(TypedDict):
    """Complete crop evaluation used as one raw tree-search return."""

    assessment: VerifierScore
    proposals: list[RankedRegion]  # Verified proposal crops, including No/Uncertain.
    raw_reward: float  # Combined verified-candidate relevance/confidence/count/diversity reward.
    candidate_checks: NotRequired[list[CandidateCheck]]


class SearchStep(TypedDict):
    """One selected frontier and its outcome; partial work never updates Q."""

    iteration: int
    node_id: int
    path: list[int]
    quality: float  # Selected branch's Q before this evaluation.
    exploration_bonus: float
    estimated_remaining_seconds: float
    selection_index: float
    calls_before: int
    calls_after: int
    completed: bool
    raw_reward: float | None  # None when the budget interrupts an evaluation.
    evaluation_error: NotRequired[str]
    elapsed_seconds: float


class SearchUsage(TypedDict):
    """Physical HTTP attempts and successful-response usage, including retries."""

    api_calls: int
    cache_hits: int
    prompt_tokens: int
    completion_tokens: int
    reported_cost_usd: (
        float  # Sum of reported charges; provider failures may omit usage.
    )
    local_score_calls: NotRequired[
        int
    ]  # Physical local Qwen generations, including repairs.
    local_prompt_calls: NotRequired[int]  # Local Qwen text-only SAM3 prompt rewrites.
    candidate_checks: NotRequired[int]  # Candidate evidence sheets verified by Qwen.
    gated_regions: NotRequired[
        int
    ]  # Regions with no SAM3 candidates whose verifier call was skipped.


class SearchResult(TypedDict):
    """Search evidence and trace; arrays are excluded from the JSON summary."""

    query: str
    config: SearchConfig
    nodes: list[SearchNode]
    trace: list[SearchStep]
    ranked_regions: list[RankedRegion]
    ranked_clusters: list[RankedCluster]
    usage: SearchUsage
    stop_reason: Literal[
        "exhausted", "api_budget", "local_budget", "iterations", "match", "error"
    ]
    error: str  # Empty on successful or budget-limited completion.
    elapsed_seconds: float
    warnings: NotRequired[
        list[str]
    ]  # Recoverable region failures; query may still finish successfully.


class NodeReport(TypedDict):
    """Serializable tree state; mask_key points into masks.npz."""

    node_id: int
    parent_id: int
    depth: int
    bbox: BoundingBox
    active_pixels: int
    children: list[int]
    visits: int
    mean_quality: float
    evaluated: bool
    exhausted: bool
    mask_key: str
    evaluation_error: NotRequired[str]


class ClusterReport(TypedDict):
    """A region-level assessment and the saved crop used to obtain it."""

    cluster_id: int
    bbox: BoundingBox
    score: float
    verifier: VerifierScore
    image_path: str
    score_components: NotRequired[RegionScoreComponents]
    scoring_input_image_path: NotRequired[
        str
    ]  # Exact unmasked region input, preserved separately from previews.


class ProposalReport(ClusterReport):
    """A localized candidate; binary mask remains in parent-crop coordinates.

    image_path saves the raw SAM3 crop. evidence_image_path saves the actual
    visual-verifier input for context-aware runs.
    """

    local_bbox: BoundingBox
    sam3_score: float | None
    cluster_score: float
    mask_key: str
    evidence_image_path: NotRequired[str]  # Lossless PNG of the actual uploaded view.
    evidence_info: NotRequired[EvidenceInfo]
    scoring_scope: NotRequired[Literal["region", "proposal"]]
    candidate_rank: NotRequired[int]
    candidate_score: NotRequired[float]
    candidate_verified: NotRequired[bool]


class SearchReport(TypedDict):
    """results.json schema; all paths are relative to the output directory."""

    query: str
    image_size: tuple[int, int]
    config: SearchConfig
    usage: SearchUsage
    stop_reason: str
    error: str
    elapsed_seconds: float
    nodes: list[NodeReport]
    trace: list[SearchStep]
    clusters: list[ClusterReport]
    proposals: list[ProposalReport]
    warnings: NotRequired[list[str]]
