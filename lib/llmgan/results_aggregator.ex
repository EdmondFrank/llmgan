defmodule Llmgan.ResultsAggregator do
  @moduledoc """
  GenServer for buffering, aggregating, and persisting test results.

  Computes metrics like:
  - Accuracy rates
  - Latency statistics
  - Token usage
  - Score distributions
  """

  use GenServer
  require Logger

  alias Llmgan.Types

  defstruct [
    :results_buffer,
    :metrics,
    :evaluations,
    :flush_interval,
    :max_buffer_size,
    :persistence_fn
  ]

  @default_flush_interval 5_000
  @default_max_buffer_size 100

  # Client API

  @doc """
  Starts the ResultsAggregator GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Adds a test result to the aggregator.
  """
  @spec add_result(map()) :: :ok
  def add_result(result) do
    GenServer.cast(__MODULE__, {:add_result, result})
  end

  @doc """
  Adds an evaluation result to the aggregator.
  """
  @spec add_evaluation(map()) :: :ok
  def add_evaluation(evaluation) do
    GenServer.cast(__MODULE__, {:add_evaluation, evaluation})
  end

  @doc """
  Gets current aggregated metrics.
  """
  @spec get_metrics() :: Types.test_metrics()
  def get_metrics do
    GenServer.call(__MODULE__, :get_metrics)
  end

  @doc """
  Gets all collected results.
  """
  @spec get_results() :: list(Types.test_result())
  def get_results do
    GenServer.call(__MODULE__, :get_results)
  end

  @doc """
  Gets all collected evaluations.
  """
  @spec get_evaluations() :: list(Types.evaluation_result())
  def get_evaluations do
    GenServer.call(__MODULE__, :get_evaluations)
  end

  @doc """
  Forces a flush of the buffer.
  """
  @spec flush() :: :ok
  def flush do
    GenServer.call(__MODULE__, :flush)
  end

  @doc """
  Resets all aggregated data.
  """
  @spec reset() :: :ok
  def reset do
    GenServer.call(__MODULE__, :reset)
  end

  @doc """
  Generates a summary report.
  """
  @spec generate_report() :: map()
  def generate_report do
    GenServer.call(__MODULE__, :generate_report)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    flush_interval = Keyword.get(opts, :flush_interval, @default_flush_interval)
    max_buffer_size = Keyword.get(opts, :max_buffer_size, @default_max_buffer_size)
    persistence_fn = Keyword.get(opts, :persistence_fn)

    # Schedule periodic flush
    schedule_flush(flush_interval)

    {:ok,
     %__MODULE__{
       results_buffer: [],
       metrics: init_metrics(),
       evaluations: [],
       flush_interval: flush_interval,
       max_buffer_size: max_buffer_size,
       persistence_fn: persistence_fn
     }}
  end

  @impl true
  def handle_cast({:add_result, result}, state) do
    new_buffer = [result | state.results_buffer]

    # Update metrics incrementally
    updated_metrics = update_metrics(state.metrics, result)

    new_state = %{state | results_buffer: new_buffer, metrics: updated_metrics}

    # Flush if buffer is full
    if length(new_buffer) >= state.max_buffer_size do
      {:noreply, do_flush(new_state)}
    else
      {:noreply, new_state}
    end
  end

  @impl true
  def handle_cast({:add_evaluation, evaluation}, state) do
    updated_evaluations = [evaluation | state.evaluations]

    # Update metrics with evaluation data
    updated_metrics = update_metrics_with_evaluation(state.metrics, evaluation)

    {:noreply, %{state | evaluations: updated_evaluations, metrics: updated_metrics}}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    {:reply, state.metrics, state}
  end

  @impl true
  def handle_call(:get_results, _from, state) do
    {:reply, Enum.reverse(state.results_buffer), state}
  end

  @impl true
  def handle_call(:get_evaluations, _from, state) do
    {:reply, Enum.reverse(state.evaluations), state}
  end

  @impl true
  def handle_call(:flush, _from, state) do
    {:reply, :ok, do_flush(state)}
  end

  @impl true
  def handle_call(:reset, _from, _state) do
    {:reply, :ok,
     %__MODULE__{
       results_buffer: [],
       metrics: init_metrics(),
       evaluations: [],
       flush_interval: @default_flush_interval,
       max_buffer_size: @default_max_buffer_size,
       persistence_fn: nil
     }}
  end

  @impl true
  def handle_call(:generate_report, _from, state) do
    report = %{
      generated_at: DateTime.utc_now(),
      metrics: state.metrics,
      summary: generate_summary(state),
      results_count: length(state.results_buffer),
      evaluations_count: length(state.evaluations)
    }

    {:reply, report, state}
  end

  @impl true
  def handle_info(:flush, state) do
    schedule_flush(state.flush_interval)
    {:noreply, do_flush(state)}
  end

  # Private Functions

  defp schedule_flush(interval) do
    Process.send_after(self(), :flush, interval)
  end

  defp init_metrics do
    %{
      total_scenarios: 0,
      completed: 0,
      passed: 0,
      failed: 0,
      errors: 0,
      total_latency_ms: 0,
      min_latency_ms: nil,
      max_latency_ms: 0,
      total_tokens: %{input: 0, output: 0},
      score_distribution: %{},
      evaluator_breakdown: %{}
    }
  end

  defp update_metrics(metrics, result) do
    latency = result.latency_ms

    %{
      metrics
      | total_scenarios: metrics.total_scenarios + 1,
        completed: if(result.success, do: metrics.completed + 1, else: metrics.completed),
        errors: if(is_nil(result.error), do: metrics.errors, else: metrics.errors + 1),
        total_latency_ms: metrics.total_latency_ms + latency,
        min_latency_ms: min(metrics.min_latency_ms || latency, latency),
        max_latency_ms: max(metrics.max_latency_ms, latency),
        total_tokens: update_token_counts(metrics.total_tokens, result.tokens_used)
    }
  end

  defp update_metrics_with_evaluation(metrics, evaluation) do
    passed_count = if(evaluation.passed, do: 1, else: 0)

    evaluator_type = evaluation.evaluator_type

    evaluator_breakdown =
      Map.update(
        metrics.evaluator_breakdown,
        evaluator_type,
        %{count: 1, passed: passed_count},
        fn existing ->
          %{
            count: existing.count + 1,
            passed: existing.passed + passed_count
          }
        end
      )

    %{
      metrics
      | passed: metrics.passed + passed_count,
        failed: metrics.failed + if(evaluation.passed, do: 0, else: 1),
        evaluator_breakdown: evaluator_breakdown
    }
  end

  defp update_token_counts(current, nil), do: current

  defp update_token_counts(current, tokens) when is_map(tokens) do
    input = Map.get(tokens, :input, 0) || Map.get(tokens, "input", 0) || 0
    output = Map.get(tokens, :output, 0) || Map.get(tokens, "output", 0) || 0

    %{
      input: current.input + input,
      output: current.output + output
    }
  end

  defp do_flush(state) do
    if is_function(state.persistence_fn) and length(state.results_buffer) > 0 do
      try do
        state.persistence_fn.(Enum.reverse(state.results_buffer))
      rescue
        e -> Logger.error("Persistence failed: #{Exception.message(e)}")
      end
    end

    %{state | results_buffer: []}
  end

  defp generate_summary(state) do
    metrics = state.metrics

    total = metrics.total_scenarios

    avg_latency =
      if total > 0 do
        metrics.total_latency_ms / total
      else
        0.0
      end

    pass_rate =
      if total > 0 do
        metrics.passed / total * 100
      else
        0.0
      end

    completion_rate =
      if total > 0 do
        metrics.completed / total * 100
      else
        0.0
      end

    %{
      pass_rate: Float.round(pass_rate, 2),
      completion_rate: Float.round(completion_rate, 2),
      error_rate: Float.round(metrics.errors / max(total, 1) * 100, 2),
      avg_latency_ms: Float.round(avg_latency, 2),
      min_latency_ms: metrics.min_latency_ms,
      max_latency_ms: metrics.max_latency_ms,
      total_tokens: metrics.total_tokens,
      evaluator_performance: calculate_evaluator_performance(metrics.evaluator_breakdown)
    }
  end

  defp calculate_evaluator_performance(evaluator_breakdown) do
    Enum.map(evaluator_breakdown, fn {type, stats} ->
      pass_rate = if(stats.count > 0, do: stats.passed / stats.count * 100, else: 0.0)

      %{
        evaluator: type,
        count: stats.count,
        pass_rate: Float.round(pass_rate, 2)
      }
    end)
  end
end
