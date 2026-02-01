defmodule Llmgan.Types do
  @moduledoc """
  Core type definitions for the LLM test scenario generation and evaluation framework.
  """

  @typedoc "A test scenario with input and expected output"
  @type scenario :: %{
    id: String.t(),
    name: String.t(),
    input: String.t() | map(),
    expected_output: String.t() | map() | nil,
    metadata: map(),
    tags: list(String.t())
  }

  @typedoc "Test result from running a scenario"
  @type test_result :: %{
    scenario_id: String.t(),
    scenario_name: String.t(),
    input: String.t() | map(),
    expected_output: String.t() | map() | nil,
    actual_output: String.t() | map(),
    latency_ms: integer(),
    tokens_used: map() | nil,
    timestamp: DateTime.t(),
    success: boolean(),
    error: String.t() | nil
  }

  @typedoc "Evaluation result with scores"
  @type evaluation_result :: %{
    scenario_id: String.t(),
    test_result: test_result(),
    scores: map(),
    passed: boolean(),
    evaluator_type: atom(),
    metadata: map()
  }

  @typedoc "Template for generating test scenarios"
  @type scenario_template :: %{
    id: String.t(),
    name: String.t(),
    prompt_template: String.t(),
    variables: list(String.t()),
    generation_strategy: atom(),
    constraints: list(term()),
    expected_output_template: String.t() | nil
  }

  @typedoc "Configuration for LLM provider"
  @type llm_config :: %{
    provider: atom(),
    model: String.t(),
    temperature: float(),
    max_tokens: integer() | nil,
    api_key: String.t() | nil,
    endpoint: String.t() | nil,
    additional_params: map()
  }

  @typedoc "Evaluation strategy configuration"
  @type evaluation_config :: %{
    strategy: :exact_match | :semantic_similarity | :llm_judge | :custom,
    threshold: float(),
    custom_fn: (String.t(), String.t() -> float()) | nil,
    llm_config: llm_config() | nil
  }

  @typedoc "Runner configuration for test execution"
  @type runner_config :: %{
    llm_config: llm_config(),
    rate_limit: integer() | nil,
    timeout_ms: integer(),
    max_retries: integer(),
    batch_size: integer()
  }

  @typedoc "Aggregated test metrics"
  @type test_metrics :: %{
    total_scenarios: integer(),
    completed: integer(),
    passed: integer(),
    failed: integer(),
    errors: integer(),
    avg_latency_ms: float(),
    total_tokens: map(),
    score_distribution: map()
  }
end
