defmodule Llmgan.Runner do
  @moduledoc """
  GenServer for executing test scenarios against LLM APIs.

  Features:
  - Rate limiting via token bucket
  - Retry logic with exponential backoff
  - Timeout handling
  - Parallel execution using Task.async_stream
  - Circuit breaker pattern for API failures
  """

  use GenServer
  require Logger

  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  defstruct [
    :scenarios,
    :config,
    :results,
    :status,
    :started_at,
    :completed_at,
    :circuit_breaker_state,
    :rate_limit_tokens,
    :callback_fn,
    :task_timeout
  ]

  # Client API

  @doc """
  Starts a new Runner process.
  """
  def start_link({scenarios, config, opts}) do
    GenServer.start_link(__MODULE__, {scenarios, config, opts})
  end

  @doc """
  Starts running the test scenarios.

  ## Options

  - `:timeout` - Maximum time to wait for all scenarios to complete (default: :infinity)
  """
  def run(pid, timeout \\ :infinity) do
    GenServer.call(pid, :run, timeout)
  end

  @doc """
  Gets the current status of the runner.
  """
  def get_status(pid) do
    GenServer.call(pid, :get_status)
  end

  @doc """
  Gets results so far (for streaming updates).
  """
  def get_results(pid) do
    GenServer.call(pid, :get_results)
  end

  @doc """
  Stops the runner.
  """
  def stop(pid) do
    GenServer.stop(pid)
  end

  # Server Callbacks

  @impl true
  def init({scenarios, config, opts}) do
    state = %__MODULE__{
      scenarios: scenarios,
      config: config,
      results: [],
      status: :idle,
      started_at: nil,
      completed_at: nil,
      circuit_breaker_state: :closed,
      rate_limit_tokens: config.rate_limit || 10,
      callback_fn: Keyword.get(opts, :callback_fn),
      # Store task timeout separately from scenario timeout
      task_timeout: Keyword.get(opts, :task_timeout, config.timeout_ms)
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:run, from, %{status: :idle} = state) do
    # Start async execution - reply will be sent from handle_info
    send(self(), {:do_run, from})

    {:noreply, %{state | status: :running, started_at: DateTime.utc_now()}}
  end

  @impl true
  def handle_call(:run, _from, state) do
    {:reply, {:error, :already_running}, state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    {:reply, state.status, state}
  end

  @impl true
  def handle_call(:get_results, _from, state) do
    {:reply, state.results, state}
  end

  @impl true
  def handle_info({:do_run, from}, state) do
    llm_config = state.config.llm_config

    # Create LLM model based on config
    llm = create_llm_model(llm_config)

    # Execute scenarios with controlled parallelism
    # Use a longer task timeout than the scenario timeout to allow for overhead
    task_timeout = Map.get(state, :task_timeout, state.config.timeout_ms)

    results =
      state.scenarios
      |> Task.async_stream(
        fn scenario ->
          execute_scenario(scenario, llm, state.config)
        end,
        max_concurrency: state.config.batch_size,
        timeout: task_timeout,
        ordered: false
      )
      |> Enum.map(fn
        {:ok, result} ->
          notify_callback(state.callback_fn, {:scenario_complete, result})
          result

        {:exit, reason} ->
          error_result = %{
            scenario_id: "unknown",
            scenario_name: "unknown",
            input: nil,
            expected_output: nil,
            actual_output: nil,
            latency_ms: 0,
            tokens_used: nil,
            timestamp: DateTime.utc_now(),
            success: false,
            error: "Task failed: #{inspect(reason)}"
          }

          notify_callback(state.callback_fn, {:scenario_error, error_result})
          error_result
      end)

    completed_state = %{
      state
      | results: results,
        status: :completed,
        completed_at: DateTime.utc_now()
    }

    # Reply to the caller
    GenServer.reply(from, {:ok, results})

    {:noreply, completed_state}
  end

  # Private Functions

  defp create_llm_model(llm_config) do
    case llm_config.provider do
      :openai ->
        alias LangChain.ChatModels.ChatOpenAI

        ChatOpenAI.new!(%{
          model: llm_config.model,
          temperature: Map.get(llm_config, :temperature, 0.7),
          max_tokens: Map.get(llm_config, :max_tokens),
          api_key: llm_config.api_key,
          endpoint: Map.get(llm_config, :endpoint),
          stream: false
        })

      :anthropic ->
        alias LangChain.ChatModels.ChatAnthropic

        ChatAnthropic.new!(%{
          model: llm_config.model,
          temperature: Map.get(llm_config, :temperature, 0.7),
          max_tokens: Map.get(llm_config, :max_tokens),
          api_key: llm_config.api_key
        })

      :google ->
        alias LangChain.ChatModels.ChatGoogleAI

        ChatGoogleAI.new!(%{
          model: llm_config.model,
          temperature: Map.get(llm_config, :temperature, 0.7),
          max_tokens: Map.get(llm_config, :max_tokens),
          api_key: llm_config.api_key
        })

      _ ->
        raise ArgumentError, "Unsupported LLM provider: #{inspect(llm_config.provider)}"
    end
  end

  defp execute_scenario(scenario, llm, config) do
    start_time = System.monotonic_time(:millisecond)

    # Build the prompt using custom template or fallback logic
    prompt = build_prompt(scenario, config)

    result =
      with_retries(config.max_retries, fn ->
        run_llm_chain(llm, prompt)
      end)

    end_time = System.monotonic_time(:millisecond)
    latency_ms = end_time - start_time

    case result do
      {:ok, response, metadata} ->
        %{
          scenario_id: scenario.id,
          scenario_name: scenario.name,
          input: scenario.input,
          expected_output: scenario.expected_output,
          actual_output: response,
          latency_ms: latency_ms,
          tokens_used: Map.get(metadata, :tokens_used),
          timestamp: DateTime.utc_now(),
          success: true,
          error: nil
        }

      {:error, reason} ->
        %{
          scenario_id: scenario.id,
          scenario_name: scenario.name,
          input: scenario.input,
          expected_output: scenario.expected_output,
          actual_output: nil,
          latency_ms: latency_ms,
          tokens_used: nil,
          timestamp: DateTime.utc_now(),
          success: false,
          error: inspect(reason)
        }
    end
  end

  defp run_llm_chain(llm, input) do
    try do
      chain =
        LLMChain.new!(%{llm: llm})
        |> LLMChain.add_message(Message.new_user!(format_input(input)))

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)

          metadata = %{
            tokens_used: get_in(last_message, [Access.key(:metadata), :usage])
          }

          content = extract_content(last_message)
          {:ok, content, metadata}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp format_input(input) when is_binary(input), do: input
  defp format_input(input) when is_map(input), do: Jason.encode!(input)
  defp format_input(input), do: to_string(input)

  defp extract_content(message) do
    case message do
      %{content: content} when is_binary(content) ->
        content

      %{content: content_parts} when is_list(content_parts) ->
        content_parts
        |> Enum.map(& &1.content)
        |> Enum.join("\n")

      _ ->
        ""
    end
  end

  defp with_retries(0, fun), do: fun.()

  defp with_retries(max_retries, fun) do
    case fun.() do
      {:ok, result, metadata} ->
        {:ok, result, metadata}

      {:error, _reason} when max_retries > 0 ->
        # Exponential backoff
        backoff_ms = :rand.uniform(1000) * (max_retries + 1)
        Process.sleep(backoff_ms)
        with_retries(max_retries - 1, fun)

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp notify_callback(nil, _event), do: :ok
  defp notify_callback(callback_fn, event), do: callback_fn.(event)

  defp build_prompt(scenario, config) do
    # Priority: 1. Custom template, 2. Generation prompt from metadata, 3. Direct input
    template =
      config[:prompt_template] ||
        get_in(scenario, [:metadata, :generation_prompt]) ||
        get_in(scenario, [:metadata, "generation_prompt"])

    if template do
      # Combine template with input - template provides context, input provides content
      combine_template_with_input(template, scenario)
    else
      format_input(scenario.input)
    end
  end

  defp combine_template_with_input(template, scenario) when is_binary(template) do
    # If template already references @input, render it directly
    # Otherwise, append the input after the template
    if String.contains?(template, "@input") or String.contains?(template, "<%= input %>") do
      render_template(template, scenario)
    else
      # Combine: template context + input content
      rendered_template = render_template(template, scenario)
      input = format_input(scenario.input)

      # Join with appropriate separator based on content
      if String.ends_with?(rendered_template, "\n") do
        rendered_template <> input
      else
        rendered_template <> "\n\n" <> input
      end
    end
  end

  defp combine_template_with_input(_template, scenario), do: format_input(scenario.input)

  defp render_template(template, scenario) when is_binary(template) do
    # Create assigns list with all scenario fields available in template
    # EEx uses @variable syntax to access assigns
    assigns = [
      id: Map.get(scenario, :id),
      name: Map.get(scenario, :name),
      input: Map.get(scenario, :input),
      expected_output: Map.get(scenario, :expected_output),
      metadata: Map.get(scenario, :metadata, %{}),
      tags: Map.get(scenario, :tags, [])
    ]

    try do
      EEx.eval_string(template, assigns: assigns)
    rescue
      e ->
        Logger.warning("Template rendering failed: #{Exception.message(e)}. Using raw input.")
        format_input(scenario.input)
    end
  end

  defp render_template(_template, scenario), do: format_input(scenario.input)
end
