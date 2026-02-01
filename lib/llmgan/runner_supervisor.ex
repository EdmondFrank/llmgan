defmodule Llmgan.RunnerSupervisor do
  @moduledoc """
  DynamicSupervisor for managing Runner processes.
  Each test batch gets its own Runner process.
  """

  use DynamicSupervisor
  require Logger

  # Client API

  @doc """
  Starts the RunnerSupervisor.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    DynamicSupervisor.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Starts a new Runner for a test batch.
  """
  def start_runner(scenarios, config, opts \\ []) do
    spec = {Llmgan.Runner, {scenarios, config, opts}}
    DynamicSupervisor.start_child(__MODULE__, spec)
  end

  @doc """
  Terminates a runner process.
  """
  def terminate_runner(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end

  @doc """
  Lists all active runner processes.
  """
  def list_runners do
    DynamicSupervisor.which_children(__MODULE__)
    |> Enum.map(fn {_, pid, _, _} -> pid end)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    DynamicSupervisor.init(
      strategy: :one_for_one,
      max_restarts: 1000,
      max_seconds: 60
    )
  end
end
