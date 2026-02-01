# LLMGan

> **Generate → Test → Evaluate**: A comprehensive LLM testing framework that uses AI to generate intelligent test scenarios, executes them in parallel with advanced retry and rate-limiting mechanisms, and evaluates responses using multiple strategies including LLM-as-judge.

LLMGan is an OTP-based testing framework built in Elixir for systematic evaluation of Large Language Models. It leverages Elixir's actor-model concurrency and supervision trees to provide resilient, scalable testing capabilities with built-in support for OpenAI, Anthropic Claude, Google AI, and local models via Ollama.

## Key Capabilities

- **🤖 AI-Powered Test Generation**: Automatically generate diverse test scenarios with expected outputs using LLMs. Simply describe what you want to test, and the framework creates comprehensive test cases covering normal cases, edge cases, and boundary conditions.

- **⚡ High-Performance Parallel Execution**: Run hundreds of test scenarios concurrently with configurable batch sizes, rate limiting, exponential backoff retry logic, and circuit breaker patterns for resilient API communication.

- **🧠 Intelligent Evaluation**: Choose from multiple evaluation strategies including exact match, semantic similarity (Levenshtein, Jaccard, cosine), LLM-as-judge for subjective quality assessment, or define custom evaluation functions.

- **📊 Comprehensive Metrics & Reporting**: Track accuracy rates, latency statistics (min/max/average), token usage across input/output, evaluator performance breakdowns, and export detailed reports for analysis.

- **🔌 Universal LLM Support**: Seamlessly works with OpenAI (GPT-4, GPT-3.5), Anthropic Claude, Google AI (Gemini), and local models through Ollama with a unified adapter interface.

- **🔄 Real-time Progress Tracking**: Execute tests asynchronously with callback functions for live progress updates, enabling integration with CI/CD pipelines and monitoring systems.

## Features

- **🎯 Scenario Generation**: Template-based, fuzzing, edge cases, and adversarial inputs
- **⚡ Parallel Execution**: Run tests concurrently with rate limiting, retry logic, and circuit breakers
- **📊 Multiple Evaluators**: Exact match, semantic similarity, LLM-as-judge, and custom functions
- **📈 Metrics & Reporting**: Comprehensive metrics on accuracy, latency, token usage, and cost
- **🔌 OpenAI-Compatible**: Works with OpenAI, Anthropic Claude, Google AI, and local models via Ollama
- **🔄 Real-time Updates**: Async execution with callbacks for progress tracking

## Installation

Add `llmgan` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:llmgan, "edmondfrank/llmgan", branch: "master"}
  ]
end
```

## Quick Start

The **Generate → Test → Evaluate** workflow:

```elixir
# 1. Configure LLM
llm_config = %{
  provider: :openai,
  model: "gpt-4",
  api_key: System.get_env("OPENAI_API_KEY"),
  temperature: 0.7
}

# 2. Generate test scenarios with LLM
{:ok, scenarios} = Llmgan.generate_scenarios(:llm, %{
  description: "Test cases for sentiment analysis",
  count: 5,
  llm_config: llm_config
})

# 3. Run tests
{:ok, results} = Llmgan.run_tests(scenarios, llm_config)

# 4. Evaluate with LLM-as-judge
eval_config = %{strategy: :llm_judge, threshold: 0.8, llm_config: llm_config}
{:ok, evaluations} = Llmgan.evaluate_results(results, eval_config)

# 5. View report
report = Llmgan.generate_report()
IO.inspect(report.summary)
```

## Architecture

```
Llmgan.Supervisor (Root)
├── Llmgan.ScenarioGenerator (GenServer)
│   └── ETS-backed template storage
├── Llmgan.RunnerSupervisor (DynamicSupervisor)
│   └── Llmgan.Runner (GenServer per batch)
├── Llmgan.EvaluatorPool (Poolboy)
│   └── Llmgan.EvaluatorWorker
└── Llmgan.ResultsAggregator (GenServer)
    └── Metrics computation & buffering
```

## Scenario Generation

### Template-based Generation

```elixir
{:ok, scenarios} = Llmgan.generate_scenarios(:template, %{
  template: "Classify sentiment: <%= text %>",
  variables_list: [
    %{text: "I love this!"},
    %{text: "This is terrible."}
  ],
  expected_output: nil
})
```

### Fuzzing

```elixir
{:ok, scenarios} = Llmgan.generate_scenarios(:fuzzing, %{
  template: "Process: <%= input %>",
  base_variables: %{type: "text"},
  fuzz_fields: ["input"],
  count: 20
})
```

### Edge Cases

```elixir
{:ok, scenarios} = Llmgan.generate_scenarios(:edge_cases, %{
  template: "Summarize: <%= content %>",
  field: "content"
})
```

### Adversarial Testing

```elixir
{:ok, scenarios} = Llmgan.generate_scenarios(:adversarial, %{
  template: "<%= input %>",
  expected_output: "Should reject or handle safely"
})
```

## Test Execution

### Basic Execution

```elixir
{:ok, results} = Llmgan.run_tests(scenarios, llm_config,
  timeout_ms: 60_000,
  max_retries: 3,
  batch_size: 5
)
```

### Async Execution with Callbacks

```elixir
callback = fn
  {:scenario_complete, result} ->
    IO.puts("✅ #{result.scenario_name}")
  {:scenario_error, result} ->
    IO.puts("❌ #{result.error}")
end

{:ok, runner_pid} = Llmgan.run_tests_async(scenarios, llm_config,
  callback_fn: callback
)

# Get results later
results = Llmgan.Runner.get_results(runner_pid)
```

## Evaluation Strategies

### Exact Match

```elixir
eval_config = %{strategy: :exact_match, threshold: 1.0}
{:ok, evaluations} = Llmgan.evaluate_results(results, eval_config)
```

### Semantic Similarity

Uses Levenshtein distance, Jaccard similarity, and cosine similarity:

```elixir
eval_config = %{
  strategy: :semantic_similarity,
  threshold: 0.8  # 0.0 to 1.0
}
{:ok, evaluations} = Llmgan.evaluate_results(results, eval_config)
```

### LLM-as-Judge

```elixir
judge_config = %{
  strategy: :llm_judge,
  threshold: 0.7,
  llm_config: %{
    provider: :openai,
    model: "gpt-4",
    api_key: System.get_env("OPENAI_API_KEY")
  }
}
{:ok, evaluations} = Llmgan.evaluate_results(results, judge_config)
```

### Custom Evaluation

```elixir
custom_fn = fn expected, actual ->
  if String.contains?(actual, expected), do: 1.0, else: 0.0
end

custom_config = %{
  strategy: :custom,
  custom_fn: custom_fn,
  threshold: 1.0
}
{:ok, evaluations} = Llmgan.evaluate_results(results, custom_config)
```

## Supported LLM Providers

| Provider | Module | Notes |
|----------|--------|-------|
| OpenAI | `Llmgan.Adapters.OpenAI` | GPT-4, GPT-3.5, etc. |
| Anthropic | `Llmgan.Adapters.Anthropic` | Claude models |
| Google AI | `Llmgan.Adapters.GoogleAI` | Gemini models |
| Ollama | `Llmgan.Adapters.Ollama` | Local models |

## Reporting

```elixir
# Generate comprehensive report
report = Llmgan.generate_report()

# Access metrics
metrics = Llmgan.get_metrics()

# Get raw results
results = Llmgan.get_results()

# Get evaluations
evaluations = Llmgan.get_evaluations()

# Reset all data
Llmgan.reset()
```

## Examples

See the `examples/` directory:

- `basic_usage.exs` - Getting started guide
- `advanced_usage.exs` - Advanced features and strategies

Run an example:

```bash
elixir examples/basic_usage.exs
```

## Configuration

Environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Documentation

Generate documentation with:

```bash
mix docs
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

