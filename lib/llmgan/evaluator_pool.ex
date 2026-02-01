defmodule Llmgan.EvaluatorPool do
  @moduledoc """
  Poolboy-based worker pool for evaluating LLM responses.
  Provides parallel evaluation capabilities with configurable pool size.
  """

  use Supervisor
  require Logger

  @default_pool_size 5
  @default_max_overflow 2

  # Client API

  @doc """
  Starts the EvaluatorPool supervisor.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    Supervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Evaluates a test result using the configured strategy.
  """
  def evaluate(test_result, evaluation_config) do
    :poolboy.transaction(
      :evaluator_pool,
      fn worker_pid ->
        GenServer.call(worker_pid, {:evaluate, test_result, evaluation_config}, 60_000)
      end,
      :infinity
    )
  end

  @doc """
  Evaluates multiple test results in parallel.
  """
  def evaluate_batch(test_results, evaluation_config) do
    test_results
    |> Task.async_stream(
      fn result ->
        evaluate(result, evaluation_config)
      end,
      max_concurrency: @default_pool_size,
      timeout: 30_000,
      ordered: false
    )
    |> Enum.map(fn
      {:ok, result} -> result
      {:exit, reason} -> {:error, reason}
    end)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    pool_size = Keyword.get(opts, :pool_size, @default_pool_size)
    max_overflow = Keyword.get(opts, :max_overflow, @default_max_overflow)

    poolboy_config = [
      name: {:local, :evaluator_pool},
      worker_module: Llmgan.EvaluatorWorker,
      size: pool_size,
      max_overflow: max_overflow
    ]

    children = [
      :poolboy.child_spec(:evaluator_pool, poolboy_config)
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end

defmodule Llmgan.EvaluatorWorker do
  @moduledoc """
  Worker process for the EvaluatorPool.
  Handles individual evaluation requests.
  """

  use GenServer
  require Logger

  alias Llmgan.Evaluators

  # Client API

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, [])
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    {:ok, %{}}
  end

  @impl true
  def handle_call({:evaluate, test_result, evaluation_config}, _from, state) do
    result = Evaluators.evaluate(test_result, evaluation_config)
    {:reply, result, state}
  end
end
