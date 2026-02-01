defmodule Llmgan do
  @moduledoc """
  LLM Test Scenario Generation and Evaluation Framework.

  An OTP-based testing framework for systematic LLM evaluation using Elixir's
  concurrency model and LangChain for prompt management.

  ## Features

  - **Scenario Generation**: Template-based, fuzzing, edge cases, adversarial inputs
  - **Parallel Execution**: Run tests concurrently with rate limiting and retry logic
  - **Multiple Evaluators**: Exact match, semantic similarity, LLM-as-judge, custom functions
  - **Metrics & Reporting**: Comprehensive metrics on accuracy, latency, and token usage
  - **OpenAI-Compatible**: Works with OpenAI, Anthropic, Google AI, and more

  ## Quick Start

      # Configure LLM
      llm_config = %{
        provider: :openai,
        model: "gpt-4",
        api_key: System.get_env("OPENAI_API_KEY"),
        temperature: 0.7
      }

      # Generate test scenarios
      {:ok, scenarios} = Llmgan.generate_scenarios(:template, %{
        template: "Translate '<%= text %>' to French",
        variables_list: [
          %{text: "Hello"},
          %{text: "Good morning"},
          %{text: "Thank you"}
        ],
        expected_output: nil
      })

      # Run tests
      {:ok, results} = Llmgan.run_tests(scenarios, llm_config)

      # Evaluate results
      eval_config = %{strategy: :semantic_similarity, threshold: 0.8}
      {:ok, evaluations} = Llmgan.evaluate_results(results, eval_config)

      # Generate report
      report = Llmgan.generate_report()
  """

  alias Llmgan.ScenarioGenerator
  alias Llmgan.RunnerSupervisor
  alias Llmgan.Runner
  alias Llmgan.EvaluatorPool
  alias Llmgan.ResultsAggregator
  alias Llmgan.Evaluators

  # Scenario Generation

  @doc """
  Generates test scenarios using the specified strategy.

  ## Strategies

  - `:template` - Template-based generation with variable substitution
  - `:fuzzing` - Fuzz testing with random variations
  - `:edge_cases` - Edge case testing (empty strings, special characters, etc.)
  - `:adversarial` - Adversarial/safety testing
  - `:llm` - Use an LLM to generate intelligent test scenarios with expected outputs

  ## Examples

      {:ok, scenarios} = Llmgan.generate_scenarios(:template, %{
        template: "Classify sentiment: <%= text %>",
        variables_list: [
          %{text: "I love this product!"},
          %{text: "This is terrible."}
        ]
      })

      {:ok, scenarios} = Llmgan.generate_scenarios(:edge_cases, %{
        template: "Process: <%= input %>",
        field: :input
      })

      # LLM-based generation with expected outputs
      llm_config = %{provider: :openai, model: "gpt-4", api_key: api_key}
      {:ok, scenarios} = Llmgan.generate_scenarios(:llm, %{
        description: "Test cases for a sentiment analysis API",
        domain: "nlp",
        count: 10,
        llm_config: llm_config
      })
  """
  @spec generate_scenarios(atom(), map()) :: {:ok, list(map())} | {:error, term()}
  def generate_scenarios(strategy, config) do
    ScenarioGenerator.generate_with_strategy(strategy, config)
  end

  @doc """
  Registers a scenario template for reuse.
  """
  @spec register_template(map()) :: :ok | {:error, term()}
  def register_template(template) do
    ScenarioGenerator.register_template(template)
  end

  @doc """
  Generates scenarios from a registered template.
  """
  @spec generate_from_template(String.t(), map(), non_neg_integer()) ::
          {:ok, list(map())} | {:error, term()}
  def generate_from_template(template_id, variables, count \\ 1) do
    ScenarioGenerator.generate_scenarios(template_id, variables, count)
  end

  # Test Execution

  @doc """
  Runs tests for the given scenarios.

  ## Options

  - `:timeout_ms` - Timeout per scenario (default: 30000)
  - `:max_retries` - Max retries on failure (default: 2)
  - `:batch_size` - Concurrent execution limit (default: 5)
  - `:rate_limit` - Rate limit tokens (default: 10)
  - `:prompt_template` - Custom EEx template for formatting prompts (optional).
    The template is combined with the scenario input to create the complete prompt.
    If template contains `@input`, input is inserted there; otherwise input is
    appended after the template. If not provided, falls back to
    scenario.metadata.generation_prompt, then to raw input directly.
  - `:task_timeout` - Timeout for Task.async_stream (defaults to `:timeout_ms`).
    Increase this if you see `Task.Supervised.stream` timeout errors.

  ## Examples

      llm_config = %{
        provider: :openai,
        model: \"gpt-4\",
        api_key: api_key
      }

      {:ok, results} = Llmgan.run_tests(scenarios, llm_config,
        timeout_ms: 60_000,
        batch_size: 3
      )

      # With custom prompt template
      {:ok, results} = Llmgan.run_tests(scenarios, llm_config,
        prompt_template: "Translate to French: <%= input %>"
      )
  """
  @spec run_tests(list(map()), map(), keyword()) :: {:ok, list(map())} | {:error, term()}
  def run_tests(scenarios, llm_config, opts \\ []) do
    runner_config = %{
      llm_config: llm_config,
      timeout_ms: Keyword.get(opts, :timeout_ms, 30_000),
      max_retries: Keyword.get(opts, :max_retries, 2),
      batch_size: Keyword.get(opts, :batch_size, 5),
      rate_limit: Keyword.get(opts, :rate_limit, 10),
      prompt_template: Keyword.get(opts, :prompt_template)
    }

    runner_opts = [
      task_timeout: Keyword.get(opts, :task_timeout, Keyword.get(opts, :timeout_ms, 600_000)),
      callback_fn: Keyword.get(opts, :callback_fn)
    ]

    # Start a runner for this batch
    case RunnerSupervisor.start_runner(scenarios, runner_config, runner_opts) do
      {:ok, runner_pid} ->
        # Execute and wait for results
        result = Runner.run(runner_pid)

        # Clean up
        RunnerSupervisor.terminate_runner(runner_pid)

        # Add results to aggregator
        with {:ok, results} <- result do
          Enum.each(results, &ResultsAggregator.add_result/1)
        end

        result

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Runs tests asynchronously with a callback for progress updates.

  ## Examples

      callback = fn
        {:scenario_complete, result} ->
          IO.puts("Completed: \#{result.scenario_name}")

        {:scenario_error, result} ->
          IO.puts("Error: \#{result.error}")
      end

      {:ok, runner_pid} = Llmgan.run_tests_async(scenarios, llm_config,
        callback_fn: callback
      )

      # Later...
      results = Llmgan.Runner.get_results(runner_pid)
  """
  @spec run_tests_async(list(map()), map(), keyword()) ::
          {:ok, pid()} | {:error, term()}
  def run_tests_async(scenarios, llm_config, opts \\ []) do
    runner_config = %{
      llm_config: llm_config,
      timeout_ms: Keyword.get(opts, :timeout_ms, 30_000),
      max_retries: Keyword.get(opts, :max_retries, 2),
      batch_size: Keyword.get(opts, :batch_size, 5),
      rate_limit: Keyword.get(opts, :rate_limit, 10),
      prompt_template: Keyword.get(opts, :prompt_template)
    }

    runner_opts = [
      task_timeout: Keyword.get(opts, :task_timeout, Keyword.get(opts, :timeout_ms, 30_000)),
      callback_fn: Keyword.get(opts, :callback_fn)
    ]

    case RunnerSupervisor.start_runner(scenarios, runner_config, runner_opts) do
      {:ok, runner_pid} ->
        # Start execution asynchronously
        Task.start(fn ->
          case Runner.run(runner_pid) do
            {:ok, results} ->
              Enum.each(results, &ResultsAggregator.add_result/1)

            _ ->
              :ok
          end
        end)

        {:ok, runner_pid}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # Evaluation

  @doc """
  Evaluates test results using the specified strategy.

  ## Strategies

  - `:exact_match` - Exact string comparison
  - `:semantic_similarity` - Levenshtein, Jaccard, and cosine similarity
  - `:llm_judge` - Use another LLM to evaluate quality
  - `:custom` - User-provided evaluation function
  - `:json_schema` - Validate JSON output conforms to schema
  - `:json_field_match` - Validate specific JSON fields match expected values

  ## Examples

      # Semantic similarity
      eval_config = %{strategy: :semantic_similarity, threshold: 0.8}
      {:ok, evaluations} = Llmgan.evaluate_results(results, eval_config)

      # LLM as judge
      judge_config = %{
        strategy: :llm_judge,
        threshold: 0.7,
        llm_config: %{provider: :openai, model: "gpt-4", api_key: key}
      }
      {:ok, evaluations} = Llmgan.evaluate_results(results, judge_config)

      # Custom function
      custom_fn = fn expected, actual ->
        if String.contains?(actual, expected), do: 1.0, else: 0.0
      end

      custom_config = %{strategy: :custom, custom_fn: custom_fn, threshold: 1.0}
      {:ok, evaluations} = Llmgan.evaluate_results(results, custom_config)

      # JSON Schema validation
      schema_config = %{
        strategy: :json_schema,
        threshold: 1.0,
        json_schema: %{
          "type" => "object",
          "properties" => %{
            "name" => %{"type" => "string"},
            "age" => %{"type" => "integer"}
          },
          "required" => ["name", "age"]
        }
      }
      {:ok, evaluations} = Llmgan.evaluate_results(results, schema_config)

      # JSON Field Matching
      field_config = %{
        strategy: :json_field_match,
        threshold: 1.0,
        field_matchers: [
          %{path: "user.name", expected: "John", match_type: :exact},
          %{path: "user.email", expected: "@example.com", match_type: :contains}
        ]
      }
      {:ok, evaluations} = Llmgan.evaluate_results(results, field_config)
  """
  @spec evaluate_results(list(map()), map()) :: {:ok, list(map())} | {:error, term()}
  def evaluate_results(results, evaluation_config) do
    evaluations =
      results
      |> Enum.map(fn result ->
        case EvaluatorPool.evaluate(result, evaluation_config) do
          {:error, reason} ->
            %{
              scenario_id: result.scenario_id,
              test_result: result,
              scores: %{error: 0.0},
              passed: false,
              evaluator_type: :error,
              metadata: %{error: inspect(reason)}
            }

          evaluation ->
            evaluation
        end
      end)

    # Add evaluations to aggregator
    Enum.each(evaluations, &ResultsAggregator.add_evaluation/1)

    {:ok, evaluations}
  end

  @doc """
  Evaluates a single result directly (bypasses the pool).
  """
  @spec evaluate(map(), map()) :: map() | {:error, term()}
  def evaluate(result, evaluation_config) do
    Evaluators.evaluate(result, evaluation_config)
  end

  # Reporting

  @doc """
  Generates a comprehensive test report.
  """
  @spec generate_report() :: map()
  def generate_report do
    ResultsAggregator.generate_report()
  end

  @doc """
  Gets current metrics.
  """
  @spec get_metrics() :: map()
  def get_metrics do
    ResultsAggregator.get_metrics()
  end

  @doc """
  Resets all aggregated data.
  """
  @spec reset() :: :ok
  def reset do
    ResultsAggregator.reset()
  end

  @doc """
  Gets all collected results.
  """
  @spec get_results() :: list(map())
  def get_results do
    ResultsAggregator.get_results()
  end

  @doc """
  Gets all collected evaluations.
  """
  @spec get_evaluations() :: list(map())
  def get_evaluations do
    ResultsAggregator.get_evaluations()
  end
end
