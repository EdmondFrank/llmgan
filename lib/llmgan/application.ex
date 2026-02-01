defmodule Llmgan.Application do
  @moduledoc """
  OTP Application for the LLM test scenario generation and evaluation framework.

  Starts the supervision tree with:
  - ScenarioGenerator: Manages test scenario templates
  - RunnerSupervisor: Dynamic supervisor for test runners
  - EvaluatorPool: Pool of workers for evaluating results
  - ResultsAggregator: Aggregates and computes metrics
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      # Scenario generator with ETS backing
      Llmgan.ScenarioGenerator,

      # Dynamic supervisor for runners
      Llmgan.RunnerSupervisor,

      # Evaluator pool with Poolboy
      Llmgan.EvaluatorPool,

      # Results aggregator for metrics
      Llmgan.ResultsAggregator
    ]

    opts = [strategy: :one_for_one, name: Llmgan.Supervisor]

    Logger.info("Starting LLM Test Framework...")
    Supervisor.start_link(children, opts)
  end
end
