# Pseudocode

This is language-neutral pseudocode for the default SAM3-first path. It
matches `algorithm_refactored/` closely but omits artifact writing, event logging,
and most validation.

## Top-level runner

```text
function run_visual_search(image, query, config, optional_models, artifact_dir):
    state = prepare_run(image, query, config, optional_models, artifact_dir)

    result = explore_tree(
        root = {bbox: (0, 0, width, height), mask: all_true},
        query,
        state.settings,
        evaluate = node -> evaluate_node(state, node),
        split    = node -> split_node(state, node),
        estimate = node -> estimate_node(state, node),
        usage    = state.usage,
    )

    result.ranked_clusters = sort(state.observed_clusters, key=score, descending)
    result.ranked_regions  = sort(
        state.observed_regions,
        key=(score, sam3_score or 0),
        descending,
    )

    save_search_result(result, image, artifact_dir)
    return result
```

## Preparation and target resolution

```text
function prepare_run(image, query, config, optional_models, artifact_dir):
    validate_search_config(config)
    require query is nonempty

    rgb = to_rgb_array(image)
    usage = zero_usage_ledger()
    context = NodeContext()

    if optional_models.scorer is absent:
        scorer = optional_models.local_scorer or load_qwen_scorer(config.device)
        if config.rewrite_sam_prompt and scorer supports resolve:
            target = scorer.resolve(query)
            target = clean_sam_prompt(target)
            if target is nonempty:
                config.sam_prompt = target
                usage.local_prompt_calls += 1
        # If resolution fails, keep config.sam_prompt or the full query.

    feature_extractor = optional_models.feature_extractor or load_dinov3()
    detector = optional_models.detector or load_sam3()

    return RunState(
        query,
        config,
        usage,
        rgb,
        scorer,
        feature_extractor,
        detector,
        caches = empty,
        observed_regions = empty,
        observed_clusters = empty,
    )
```

## Tree expansion

```text
function new_node(region, node_id, parent_id, depth):
    return {
        bbox: region.bbox,
        mask: region.mask,
        node_id, parent_id, depth,
        children: [],
        visits: 0,
        reward_sum: 0.0,
        evaluated: False,
        exhausted: False,
    }

function expand(node, nodes, config, split):
    if node.depth >= config.max_depth:
        node.exhausted = True
        emit split_skipped
        return

    children = split(node)                  # cached per node
    for region in children:
        child_id = len(nodes)
        nodes.append(new_node(region, child_id, node.node_id, node.depth + 1))
        node.children.append(child_id)

    node.exhausted = (node.children is empty)
    emit children_created
```

## Cost-aware UCT

```text
function remaining_cost(node, nodes, estimate):
    active = [nodes[i] for i in node.children if not nodes[i].exhausted]

    if active is empty:
        return max(0.0, estimate(node))

    n = len(active)
    denominator = sum(child.visits for child in active) + 1
    total = 0.0
    for child in active:
        weight = (child.visits + 1 / n) / denominator
        total += weight * remaining_cost(child, nodes, estimate)
    return total

function branch_terms(child, parent, nodes, config, estimate):
    if child.visits == 0:
        quality = 0.0
        bonus = 0.0
    else:
        quality = child.reward_sum / child.visits
        bonus = 0.0
        if config.strategy != "greedy":
            bonus = config.exploration_weight
                    * sqrt(log(parent.visits + 1) / child.visits)

    cost = remaining_cost(child, nodes, estimate)
    penalty = 0.0
    if config.strategy == "agent-uct":
        penalty = config.cost_weight * cost / config.reference_cost_seconds

    index = quality + bonus - penalty
    return {
        quality,
        exploration_bonus: bonus,
        estimated_remaining_seconds: cost,
        selection_index: index,
    }

function select_child(parent, nodes, config, estimate):
    active = [nodes[i] for i in parent.children if not nodes[i].exhausted]
    require active is nonempty

    unvisited = [child for child in active if child.visits == 0]
    pool = unvisited if unvisited is nonempty else active

    terms = {child.node_id: branch_terms(child, parent, ...) for child in pool}

    # Highest index; ties go to the lowest node ID.
    chosen = argmax(
        pool,
        key = child -> (terms[child].selection_index, -child.node_id),
    )
    emit branch_selected
    return chosen.node_id, terms[chosen.node_id]

function select_path(nodes, config, estimate):
    path = [0]
    first_level_terms = {
        quality: 0.0,
        exploration_bonus: 0.0,
        estimated_remaining_seconds: estimate(nodes[0]),
        selection_index: 0.0,
    }

    while nodes[path.last].children is nonempty:
        child_id, terms = select_child(nodes[path.last], nodes, config, estimate)
        if len(path) == 1:
            first_level_terms = terms
        path.append(child_id)

    return path, first_level_terms
```

## Search loop

```text
function explore_tree(root, query, config, evaluate, split, estimate, usage):
    validate_search_config(config)
    nodes = [new_node(root, 0, -1, 0)]
    expand(nodes[0], nodes, config, split)

    if nodes[0].children is empty:
        # A non-partitionable image is still evaluated once.
        nodes[0].exhausted = False

    for iteration in 1 .. config.max_iterations:
        if nodes[0].exhausted:
            return "exhausted"

        if config.max_api_calls > 0 and usage.api_calls >= config.max_api_calls:
            return "api_budget"

        path, first_terms = select_path(nodes, config, estimate)
        node = nodes[path.last]
        emit iteration_started(node, path)

        step = start_step(first_terms, iteration, node, path, usage.api_calls)

        try:
            evidence = evaluate(node)

        except BudgetExceeded as error:
            # Partial work is retained, but no reward is backed up.
            step.completed = False
            step.raw_reward = None
            finish_step(step)
            append step to trace
            return "local_budget" if error is LocalBudgetExceeded else "api_budget"

        except InvalidScoreResponse as error:
            # Malformed verifier output is uncertainty, not a zero reward.
            node.evaluation_error = error
            step.evaluation_error = error
            finish_step(step)
            expand(node, nodes, config, split)
            refresh_exhaustion(nodes, path)
            append step to trace
            continue

        except (RuntimeError, ValueError) as error:
            # Fatal execution error; keep earlier evidence.
            finish_step(step)
            append step to trace
            return "error", error

        node.assessment = evidence.assessment
        node.evaluated = True
        reward = evidence.raw_reward
        require reward is finite and 0 <= reward <= 1

        backup_reward(nodes, path, reward)
        step.completed = True
        step.raw_reward = reward
        finish_step(step)

        if config.stop_on_match and match_found(evidence, config):
            append step to trace
            return "match"

        if config.max_api_calls > 0 and usage.api_calls >= config.max_api_calls:
            append step to trace
            return "api_budget"

        expand(node, nodes, config, split)
        refresh_exhaustion(nodes, path)
        append step to trace

    return "exhausted" if nodes[0].exhausted else "iterations"
```

## Backup and exhaustion

```text
function backup_reward(nodes, path, reward):
    for node_id in path:
        node = nodes[node_id]
        node.visits += 1
        node.reward_sum += reward
        emit reward_backed_up(node, reward)

function refresh_exhaustion(nodes, path):
    for node_id in reverse(path excluding the leaf):
        node = nodes[node_id]
        node.exhausted = node.children is nonempty
                         and all(nodes[child].exhausted for child in node.children)
```

## Node evaluation (default SAM3-first path)

```text
function evaluate_node(state, node):
    pixels = state.rgb[node.bbox]
    save_node_input(pixels, node)

    if state.settings.sam3_first:
        return evaluate_candidates(state, node, pixels)
    else:
        return evaluate_legacy(state, node, pixels)
```

```text
function evaluate_candidates(state, node, pixels):
    detections = detect_region(state, node, pixels)
    distinct = distinct_detections(detections, state.settings)

    if distinct is empty:
        state.usage.gated_regions += 1
        emit region_gated("no_candidates")
        components = zero_reward_components(gate = "no_candidates")
        return evidence(reward = 0, proposals = empty)

    if state.settings.verification_mode == "region"
       and state.local_scorer supports score_candidates:
        checks = verify_region_candidates(state, node, distinct, pixels)
        if checks is null:
            checks = verify_candidates(state, node, distinct)
    else:
        checks = verify_candidates(state, node, distinct)

    components = aggregate_verified_region(distinct, checks, state.settings)
    components.raw_box_count = len(detections)
    components.unique_box_count = len(distinct)

    assessment = best_checked_rating(checks, distinct)
    proposals = build_proposals(node, distinct, checks, components)
    register_proposals(state, proposals)

    return evidence(
        assessment,
        proposals,
        reward = components.combined_score,
        candidate_checks = checks,
    )
```

```text
function detect_region(state, node, pixels):
    if node.node_id in state.detection_cache:
        return state.detection_cache[node.node_id]

    prompt = state.settings.sam_prompt or state.query
    raw = state.detector(pixels, prompt)

    kept = [
        detection for detection in raw
        if detection.sam3_score is None
           or detection.sam3_score > state.settings.sam3_threshold
    ]
    kept = sort(kept, key=sam3_score, descending)
    state.detection_cache[node.node_id] = kept
    return kept
```

## Verification and reward

```text
function verify_region_candidates(state, node, distinct, pixels):
    limit = state.settings.region_candidate_limit
    if limit <= 0 or distinct is empty:
        return []

    subset = distinct[0 : limit]
    candidates = [
        {
            index: i + 1,
            label: state.settings.sam_prompt or state.query,
            confidence: detection.sam3_score,
            bbox: detection.bbox,
        }
        for i, detection in enumerate(subset)
    ]

    sheet = draw_numbered_boxes(pixels, candidates)
    ratings = state.local_scorer.score_candidates(
        sheet,
        query = state.query,
        candidates = candidates,
        on_request = state.charge_local,
    )

    checks = []
    for i, detection in enumerate(subset):
        rating = ratings[i + 1]
        if rating is missing:
            continue
        checks.append({
            rank: i + 1,
            detection_index: i,
            local_bbox: detection.bbox,
            bbox: to_global(node.bbox, detection.bbox),
            sam3_score: detection.sam3_score,
            candidate_score: candidate_score(
                rating.score if rating.answer != "No" else None,
                detection.sam3_score,
            ),
            qwen: rating,
        })
    return checks
```

```text
function verify_candidates(state, node, distinct):
    limit = state.settings.candidate_check_limit
    accept = state.settings.candidate_accept_score
    checks = []

    for index, detection in enumerate(distinct[0 : limit]):
        evidence_sheet = build_context_detail_sheet(
            state.source_image,
            to_global(node.bbox, detection.bbox),
        )
        rating = score_crop(evidence_sheet, state.query, "context_detail")
        check = make_check(detection, rating, evidence_sheet)
        checks.append(check)

        if rating.answer == "Yes" and rating.score >= accept:
            break

    return checks
```

```text
function candidate_score(qwen_score, sam3_score):
    qwen = qwen_score or 0.0
    sam3 = sam3_score or 0.0
    return (0.7 * qwen + 0.3 * sam3) / (0.7 + 0.3)

function aggregate_verified_region(detections, checks, config):
    best_qwen = max(
        check.qwen.score
        for check in checks
        if check.qwen exists and check.qwen.answer != "No"
        default 0.0
    )

    verified = [
        check for check in checks
        if check.qwen exists
           and check.qwen.answer == "Yes"
           and check.qwen.score >= config.candidate_accept_score
    ]

    if verified is empty:
        combined = 0.0
        confidence = 0.0
        count = 0.0
        overlap = 0.0
        diversity = 0.0
    else:
        boxes = [detections[check.detection_index] for check in verified]
        confidence = mean(box.sam3_score or 0.0 for box in boxes)
        pairs = [iou(a.bbox, b.bbox) for a, b in combinations(boxes, 2)]
        overlap = mean(pairs) if pairs is nonempty else 0.0
        count = min(len(boxes), config.region_box_count_cap)
                / config.region_box_count_cap
        diversity = 1.0 - overlap
        combined = (
            0.65 * best_qwen
          + 0.20 * confidence
          + 0.10 * count
          + 0.05 * diversity
        ) / (0.65 + 0.20 + 0.10 + 0.05)

    gate = "no_candidates" if detections is empty
           else "budget" if checks is empty
           else "unverified" if verified is empty
           else "verified"

    return {
        qwen_score: best_qwen,
        mean_confidence: confidence,
        count_score: count,
        mean_pairwise_iou: overlap,
        diversity_score: diversity,
        combined_score: combined,
        gate,
    }
```

## Splitting

```text
function split_node(state, node):
    if node.node_id in state.split_cache:
        return state.split_cache[node.node_id]

    regions = semantic_children(
        state.rgb,
        node,
        state.settings,
        state.feature_extractor,
    )
    state.split_cache[node.node_id] = regions
    return regions
```

```text
function semantic_children(image, node, config, extract_features):
    if node.depth >= config.max_depth
       or active_pixels(node) < config.min_region_pixels
       or min(node.mask.shape) < 32:
        return []

    # DINO/SLIC operate on the masked crop, not the raw rectangular crop.
    pixels = render_region_with_black_outside_mask(image, node)
    feature_map = extract_features(pixels)               # DINOv3 -> 3 PCA channels
    feature_map = resize_to_max_side(feature_map, config.segmentation_max_side)

    segmentation = choose_slic_by_mean_entropy(
        feature_map,
        config.slic,
        config.n_segments_values,
        config.entropy_bins,
    )

    n_clusters = min(config.branching_factor, len(segmentation.region_ids))
    if n_clusters < 2:
        return spatial_quadrants(node)

    clusters = kmeans(
        features = mean_pca_color_per_superpixel
                   + config.cluster_spatial_weight * normalized_centroid,
        k = n_clusters,
    )

    children = []
    for cluster_id in unique(clusters.labels[node.mask]):
        mask = (clusters.labels == cluster_id) & node.mask
        if mask is empty or mask == node.mask:
            continue
        children.append(tightened_child(mask, node))

    if len(children) < 2
       or no_child_is_smaller_than_90_percent_of_parent_area(children)
       or any(poor_geometry(child) for child in children):
        return spatial_quadrants(node)

    return children
```

```text
function spatial_quadrants(node):
    children = []
    for each quadrant tile of node.bbox:
        mask = node.mask restricted to the quadrant
        if mask has pixels and mask is not the whole parent mask:
            children.append({bbox: full_tile_bbox, mask: mask})
    return children
```

## Cost estimate

```text
function estimate_node(state, node):
    pending = state.detection_cache.get(node.node_id)   # None = not detected yet

    cost = 0.0
    if pending is None:
        cost += state.mean_cost("detect", default = 1.0)

    if state.settings.sam3_first:
        if state.settings.verification_mode == "region":
            can_check = state.settings.region_candidate_limit > 0
            checks = 1 if can_check
                        and node not in state.checked_cache
                        and pending != []
                     else 0
        else:
            limit = state.settings.candidate_check_limit
            if pending is None:
                checks = max(1, min(limit, 2))
            elif pending is empty:
                checks = 0
            elif node in state.checked_cache:
                checks = 0
            else:
                checks = min(limit, len(pending))

        cost += checks * state.mean_cost("score", default = 3.0)
    else:
        if region_view_not_cached(node):
            cost += state.mean_cost("score", default = 3.0)

    if node.depth < state.settings.max_depth
       and node not in state.split_cache:
        cost += state.mean_cost("split", default = 1.0)

    return cost
```

## Key defaults

```text
strategy                 = "agent-uct"
max_api_calls            = 32
max_iterations           = 100
max_depth                = 3
branching_factor         = 5
min_region_pixels        = 256
max_proposals_per_node   = 16
exploration_weight       = 1.4
cost_weight              = 0.2
reference_cost_seconds   = 10.0
sam3_threshold           = 0.1
mask_threshold           = 0.5
sam3_first               = True
verification_mode        = "region"
region_candidate_limit   = 5
candidate_check_limit    = 2
candidate_accept_score   = 0.5
stop_on_match            = False
match_threshold          = 0.9
max_local_scores         = 32
```

The defaults are copied from `search/config.py:8`; use that function as the
source of truth.
