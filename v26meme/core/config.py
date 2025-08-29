# v26meme/core/config.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class SystemConfig(BaseModel):
    loop_interval_seconds: int
    log_level: str
    redis_host: str
    redis_port: int
    seed: Optional[int] = None

class FeaturesConfig(BaseModel):
    zscore_lookback: int
    volatility_windows: List[int]
    parkinson_vol_windows: List[int]
    momentum: Dict[str, List[int]]

class DataSourceConfig(BaseModel):
    exchanges: List[str]

class HarvesterLimitsConfig(BaseModel):
    ohlcv_limit: Optional[int] = None

class AggregateTimeframesConfig(BaseModel):
    enabled: bool
    from_tf: str = Field(..., alias='from')
    to: List[str]
    min_native_rows_threshold: int
    max_tail_hours: int

class HarvesterCheckpointReconcileConfig(BaseModel):
    enabled: bool

class HarvesterConfig(BaseModel):
    core_symbols: List[str]
    restrict_single_symbol: bool
    dynamic_enabled: bool
    timeframes_by_lane: Dict[str, List[str]]
    partial_harvest: bool
    aggregate_timeframes: AggregateTimeframesConfig
    checkpoint_reconcile: HarvesterCheckpointReconcileConfig
    bootstrap_mode: str
    min_coverage_for_research: float
    high_coverage_threshold: float
    priority_symbols: List[str]
    panel_target_days: Dict[str, int]
    bootstrap_days_default: Dict[str, int]
    max_flat_fill_bars: int
    limits_by_venue: Dict[str, HarvesterLimitsConfig]

class FeedsConfig(BaseModel):
    cryptopanic: Dict[str, bool]
    onchain: Dict[str, Any]
    orderflow: Dict[str, Any]

class ValidationDSRConfig(BaseModel):
    enabled: bool
    min_prob: float
    benchmark_sr: float

class ValidationSparseTradesConfig(BaseModel):
    enabled: bool
    winrate_p_threshold_trades: int

class ValidationBootstrapConfig(BaseModel):
    enabled: bool
    n_iter: int
    min_trades: int
    seed: int
    method: str

class ValidationConfig(BaseModel):
    dsr: ValidationDSRConfig
    sparse_trades: ValidationSparseTradesConfig
    bootstrap: ValidationBootstrapConfig

class DiscoveryGateStageEscalationConfig(BaseModel):
    survivor_density_min: float
    median_trades_min: int
    patience_cycles: int

class DiscoveryStagnationConfig(BaseModel):
    enabled: bool
    window: int
    min_pval_delta: float
    mutation_bump: float

class DiscoveryFitnessWeightsConfig(BaseModel):
    profit_signal: float
    trade_signal: float
    complexity_penalty: Optional[float] = 0.0

class DiscoveryPrefilterConfig(BaseModel):
    enabled: bool = True
    sign_p_max: float = 0.20
    median_bps_min: float = 1.0

class DiscoveryNoveltyConfig(BaseModel):
    enabled: bool = True
    max_per_family: int = 1
    include_depth: bool = True

# NEW: promotion criteria sub-config to align schema with config.yaml usage
class PromotionCriteriaConfig(BaseModel):
    min_trades: Optional[int] = None
    min_sortino: Optional[float] = None
    min_sharpe: Optional[float] = None
    min_win_rate: Optional[float] = None
    max_mdd: Optional[float] = None

class DiscoveryConfig(BaseModel):
    max_promotions_per_cycle: int
    max_promotions_per_day: int
    base_features: List[str]
    min_panel_symbols: int
    min_bars_per_symbol: int
    panel_symbols: int
    feature_min_non_nan_ratio: float
    feature_min_variance: float
    max_workers: int
    population_size: int
    generations_per_cycle: int
    cv_folds: int
    cv_embargo_bars: int
    fdr_alpha: float
    survivor_top_k: int
    gate_stage_escalation: DiscoveryGateStageEscalationConfig
    factor_correlation_max: float
    enforce_current_gates_on_start: bool
    max_return_padding_trim: int
    max_population_size: int
    reseed_fraction: float
    fitness_drawdown_penalty_scale: float
    rejection_sample_size: int
    rejection_alert_ratio: float
    fitness_concentration_penalty_scale: float
    fitness_variance_penalty_scale: float
    fitness_activation_scale: int
    pbo_splits: int
    pbo_min_symbols: int
    stagnation: DiscoveryStagnationConfig
    debug_relaxed_gates: bool
    fitness_weights: DiscoveryFitnessWeightsConfig
    promotion_criteria: Optional[PromotionCriteriaConfig] = None  # <-- added
    prefilter: Optional[DiscoveryPrefilterConfig] = DiscoveryPrefilterConfig()
    novelty: Optional[DiscoveryNoveltyConfig] = DiscoveryNoveltyConfig()

class ProberConfig(BaseModel):
    enabled: bool
    perturbations: int
    delta_fraction: float
    min_robust_score: float

class LanesOverridesConfig(BaseModel):
    kelly_fraction: float

class LanesConfig(BaseModel):
    smoothing_factor: float
    max_step_change: float
    initial_weights: Dict[str, float]
    overrides: Dict[str, LanesOverridesConfig]

class EnsembleConfig(BaseModel):
    enabled: bool
    min_alphas_for_ensemble: int
    max_alphas_in_ensemble: int
    weighting_scheme: str

class PortfolioConfig(BaseModel):
    max_alpha_concentration: float
    kelly_fraction: float
    min_allocation_weight: float

class RiskPhaseConfig(BaseModel):
    kelly_fraction: float
    max_order_notional_usd: int

class RiskConserveModeConfig(BaseModel):
    dd_trigger_pct: float
    gross_weight_scalar: float
    kelly_scalar: float

class RiskConfig(BaseModel):
    equity_floor_pct: float
    daily_stop_pct: float
    max_symbol_weight: float
    max_gross_weight: float
    max_order_notional_usd: int
    max_consecutive_errors: int
    phases: Dict[str, RiskPhaseConfig]
    conserve_mode: RiskConserveModeConfig

class ExecutionConfig(BaseModel):
    mode: str
    primary_exchange: str
    trade_universe_source: str
    paper_fees_bps: int
    paper_slippage_bps: int
    max_order_notional_usd: int

class LLMAdaptiveTempConfig(BaseModel):
    enabled: bool
    rejection_window: int
    rejection_threshold: float
    temp_increment: float
    temp_max: float

class LLMFeatureSuppressionConfig(BaseModel):
    enabled: bool
    min_sharpe_for_inclusion: float
    min_trades_for_inclusion: int
    lookback_cycles: int

class LLMJsonHardenConfig(BaseModel):
    enabled: bool
    max_retries: int
    retry_delay_seconds: int

class LLMTelemetryConfig(BaseModel):
    log_prompts: bool
    log_tokens: bool
    log_latency: bool

class LLMGuardrailsConfig(BaseModel):
    max_prompt_tokens: int
    max_completion_tokens: int
    concentration_threshold_single: float
    concentration_threshold_total: float
    variance_threshold_min: float
    variance_threshold_max: float

class LLMConfig(BaseModel):
    provider: str
    enabled: bool
    proposer_enabled: bool
    model: str
    temperature: float
    max_tokens: int
    max_suggestions_per_cycle: int
    threshold_min: float
    threshold_max: float
    adaptive_temp: LLMAdaptiveTempConfig
    feature_suppression: LLMFeatureSuppressionConfig
    json_harden: LLMJsonHardenConfig
    telemetry: LLMTelemetryConfig
    guardrails: LLMGuardrailsConfig

class EILConfig(BaseModel):
    enabled: bool
    fast_window_days: int
    max_parallel_jobs: int
    max_queue: int
    survivor_top_k: int
    scan_batch_size: int
    max_cycles: int
    timeframe_preference: List[str] = []

class AdaptiveConfig(BaseModel):
    enabled: bool
    stop_vol_window_bars: int
    daily_stop_pct_floor: float
    daily_stop_pct_ceiling: float
    stop_vol_multiplier: float
    screener_max_markets_min: int
    screener_max_markets_max: int
    population_size_min: int
    population_size_max: int

class RegistryConfig(BaseModel):
    allowed_quotes_global: List[str]
    allowed_quotes_by_venue: Dict[str, List[str]]
    base_aliases: Dict[str, List[str]]
    cache_ttl_s: int
    catalog_refresh_seconds: int
    timeframe_aliases_by_venue: Dict[str, Dict[str, str]]

class ScreenerConfig(BaseModel):
    snapshot_dir: str
    min_24h_volume_usd: int
    min_avg_daily_volume_14d: int
    min_trades_per_day_14d: int
    max_bid_ask_spread_pct: float
    min_start_date_days_ago: int
    min_price: float
    max_markets: int
    typical_order_usd: int
    max_spread_bps: int
    max_impact_bps: int
    stablecoin_parity_warn_bps: int
    derivatives_enabled: bool
    sentiment_weight: float
    exclude_stable_stable: bool
    quotas: Optional[Dict[str, Dict[str, int]]] = None

class RootConfig(BaseModel):
    system: SystemConfig
    features: FeaturesConfig
    data_source: DataSourceConfig
    harvester: HarvesterConfig
    feeds: FeedsConfig
    validation: ValidationConfig
    discovery: DiscoveryConfig
    prober: ProberConfig
    lanes: LanesConfig
    ensemble: EnsembleConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    execution: ExecutionConfig
    llm: LLMConfig
    eil: EILConfig
    adaptive: AdaptiveConfig
    registry: RegistryConfig
    screener: ScreenerConfig
