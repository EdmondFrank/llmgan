#!/usr/bin/env elixir

# Basic Usage Example for LLM Test Framework
#
# This example demonstrates:
# 1. Generating test scenarios from templates
# 2. Running tests against an LLM
# 3. Evaluating results with semantic similarity
# 4. Generating a report

Mix.install([
  {:llmgan, path: ".."}
])

require Logger

# Ensure the application is started
Application.ensure_all_started(:llmgan)

IO.puts("=" |> String.duplicate(60))
IO.puts("LLM Test Framework - Basic Usage Example")
IO.puts("=" |> String.duplicate(60))

# ============================================================================
# Step 1: Generate Test Scenarios
# ============================================================================

IO.puts("\n📋 Step 1: Generating test scenarios...")

# Generate scenarios using template strategy
{:ok, scenarios} = Llmgan.generate_scenarios(:template, %{
  template: "Translate the following English text to French: '<%= text %>'",
  variables_list: [
    %{text: "Hello, how are you?"},
    %{text: "Good morning"},
    %{text: "Thank you very much"},
    %{text: "What is your name?"},
    %{text: "I love programming"}
  ],
  expected_output: nil,
  tags: ["translation", "french"]
})

IO.puts("✅ Generated #{length(scenarios)} test scenarios")

# Display first few scenarios
Enum.take(scenarios, 3) |> Enum.each(fn s ->
  IO.puts("  - #{s.name}: #{String.slice(s.input, 0, 50)}...")
end)

# ============================================================================
# Step 2: Configure LLM (using mock for demo)
# ============================================================================

IO.puts("\n🤖 Step 2: Configuring LLM...")

# For this demo, we'll use a mock configuration
# In production, use actual API keys
llm_config = %{
  provider: :openai,
  model: "gpt-4",
  api_key: System.get_env("OPENAI_API_KEY") || "demo-key",
  temperature: 0.7,
  max_tokens: 100
}

IO.puts("✅ LLM configured (provider: #{llm_config.provider}, model: #{llm_config.model})")

# ============================================================================
# Step 3: Run Tests
# ============================================================================

IO.puts("\n🧪 Step 3: Running tests...")

# Note: In a real scenario, this would call the actual LLM API
# For demo purposes, we'll simulate results

mock_results =
  scenarios
  |> Enum.with_index()
  |> Enum.map(fn {scenario, idx} ->
    # Simulate different responses
    responses = [
      "Bonjour, comment allez-vous?",
      "Bonjour",
      "Merci beaucoup",
      "Comment vous appelez-vous?",
      "J'adore programmer"
    ]

    %{
      scenario_id: scenario.id,
      scenario_name: scenario.name,
      input: scenario.input,
      expected_output: scenario.expected_output,
      actual_output: Enum.at(responses, idx, "Translated text"),
      latency_ms: :rand.uniform(1000) + 500,
      tokens_used: %{input: 20, output: 15},
      timestamp: DateTime.utc_now(),
      success: true,
      error: nil
    }
  end)

# Add mock results to aggregator
Enum.each(mock_results, &Llmgan.ResultsAggregator.add_result/1)

IO.puts("✅ Completed #{length(mock_results)} test runs")

# ============================================================================
# Step 4: Evaluate Results
# ============================================================================

IO.puts("\n📊 Step 4: Evaluating results...")

# Evaluate with semantic similarity
eval_config = %{
  strategy: :semantic_similarity,
  threshold: 0.6
}

# For demo, create some expected outputs to compare
mock_results_with_expected =
  mock_results
  |> Enum.with_index()
  |> Enum.map(fn {result, idx} ->
    expected = [
      "Bonjour, comment allez-vous?",
      "Bonjour",
      "Merci beaucoup",
      "Quel est votre nom?",  # Slightly different
      "J'aime la programmation"  # Different phrasing
    ]

    %{result | expected_output: Enum.at(expected, idx)}
  end)

{:ok, evaluations} = Llmgan.evaluate_results(mock_results_with_expected, eval_config)

IO.puts("✅ Evaluated #{length(evaluations)} results")

# Show sample evaluation
sample = List.first(evaluations)
if sample do
  IO.puts("\n📋 Sample evaluation:")
  IO.puts("  Scenario: #{sample.scenario_name}")
  IO.puts("  Expected: #{sample.test_result.expected_output}")
  IO.puts("  Actual:   #{sample.test_result.actual_output}")
  IO.puts("  Score:    #{Float.round(sample.scores.overall, 2)}")
  IO.puts("  Passed:   #{if(sample.passed, do: "✅", else: "❌")}")
end

# ============================================================================
# Step 5: Generate Report
# ============================================================================

IO.puts("\n📈 Step 5: Generating report...")

report = Llmgan.generate_report()

IO.puts("\n" <> "=" |> String.duplicate(60))
IO.puts("TEST REPORT")
IO.puts("=" |> String.duplicate(60))

summary = report.summary

IO.puts("""

Metrics:
  • Total Scenarios: #{report.metrics.total_scenarios}
  • Completed:       #{report.metrics.completed}
  • Passed:          #{report.metrics.passed}
  • Failed:          #{report.metrics.failed}
  • Errors:          #{report.metrics.errors}

Performance:
  • Pass Rate:       #{summary.pass_rate}%
  • Completion Rate: #{summary.completion_rate}%
  • Error Rate:      #{summary.error_rate}%
  • Avg Latency:     #{summary.avg_latency_ms}ms
  • Min Latency:     #{summary.min_latency_ms || "N/A"}ms
  • Max Latency:     #{summary.max_latency_ms}ms

Token Usage:
  • Input Tokens:  #{summary.total_tokens.input}
  • Output Tokens: #{summary.total_tokens.output}
""")

IO.puts("=" |> String.duplicate(60))
IO.puts("✨ Example completed!")
IO.puts("=" |> String.duplicate(60))

# Reset for clean state
Llmgan.reset()
