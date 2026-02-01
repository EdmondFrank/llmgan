#!/usr/bin/env elixir

# Advanced Usage Example for LLM Test Framework
#
# This example demonstrates:
# 1. Multiple generation strategies (fuzzing, edge cases, adversarial)
# 2. Custom evaluation functions
# 3. Async test execution with callbacks
# 4. Multiple evaluators with different thresholds
# 5. LLM-as-judge evaluation
# 6. Exporting results

Mix.install([
  {:llmgan, path: ".."}
])

require Logger

Application.ensure_all_started(:llmgan)

IO.puts("=" |> String.duplicate(70))
IO.puts("LLM Test Framework - Advanced Usage Example")
IO.puts("=" |> String.duplicate(70))

# ============================================================================
# Part 1: Edge Case Testing
# ============================================================================

IO.puts("\n🔍 Part 1: Edge Case Testing")
IO.puts("-" |> String.duplicate(70))

{:ok, edge_cases} = Llmgan.generate_scenarios(:edge_cases, %{
  template: "Summarize the following text: <%= input %>",
  field: "input",
  tags: ["edge_cases"]
})

IO.puts("✅ Generated #{length(edge_cases)} edge case scenarios")

edge_cases
|> Enum.take(3)
|> Enum.each(fn scenario ->
  input_preview =
    scenario.input
    |> String.slice(0, 40)
    |> String.replace("\n", "\\n")

  IO.puts("  • #{scenario.name}")
  IO.puts("    Input: #{input_preview}...")
end)

# ============================================================================
# Part 2: Fuzzing Test
# ============================================================================

IO.puts("\n🎲 Part 2: Fuzzing Test")
IO.puts("-" |> String.duplicate(70))

{:ok, fuzz_scenarios} = Llmgan.generate_scenarios(:fuzzing, %{
  template: "Classify sentiment: <%= text %>",
  base_variables: %{category: "review"},
  fuzz_fields: ["text"],
  count: 5,
  tags: ["fuzzing"]
})

IO.puts("✅ Generated #{length(fuzz_scenarios)} fuzz test scenarios")

fuzz_scenarios
|> Enum.take(3)
|> Enum.each(fn scenario ->
  IO.puts("  • #{scenario.name}")
  IO.puts("    Input: #{String.slice(scenario.input, 0, 50)}...")
end)

# ============================================================================
# Part 3: Adversarial Testing
# ============================================================================

IO.puts("\n🛡️ Part 3: Adversarial/Safety Testing")
IO.puts("-" |> String.duplicate(70))

{:ok, adversarial_scenarios} = Llmgan.generate_scenarios(:adversarial, %{
  template: "User input: <%= input %>",
  expected_output: "Should reject or handle safely",
  tags: ["safety", "adversarial"]
})

IO.puts("✅ Generated #{length(adversarial_scenarios)} adversarial scenarios")

adversarial_scenarios
|> Enum.each(fn scenario ->
  IO.puts("  • #{scenario.name}")
end)

# ============================================================================
# Part 4: Custom Evaluation
# ============================================================================

IO.puts("\n⚙️ Part 4: Custom Evaluation Function")
IO.puts("-" |> String.duplicate(70))

# Create mock results for sentiment classification
sentiment_results = [
  %{
    scenario_id: "s1",
    scenario_name: "Positive review",
    input: "I love this product!",
    expected_output: "positive",
    actual_output: "positive",
    latency_ms: 500,
    tokens_used: %{input: 10, output: 2},
    timestamp: DateTime.utc_now(),
    success: true,
    error: nil
  },
  %{
    scenario_id: "s2",
    scenario_name: "Negative review",
    input: "Terrible experience",
    expected_output: "negative",
    actual_output: "negative",
    latency_ms: 520,
    tokens_used: %{input: 10, output: 2},
    timestamp: DateTime.utc_now(),
    success: true,
    error: nil
  },
  %{
    scenario_id: "s3",
    scenario_name: "Neutral review",
    input: "It's okay",
    expected_output: "neutral",
    actual_output: "positive",  # Wrong!
    latency_ms: 480,
    tokens_used: %{input: 8, output: 2},
    timestamp: DateTime.utc_now(),
    success: true,
    error: nil
  }
]

# Custom evaluator: exact match for classification
custom_classifier_fn = fn expected, actual ->
  if String.downcase(expected) == String.downcase(actual) do
    1.0
  else
    0.0
  end
end

custom_config = %{
  strategy: :custom,
  custom_fn: custom_classifier_fn,
  threshold: 1.0
}

{:ok, classifications} = Llmgan.evaluate_results(sentiment_results, custom_config)

IO.puts("✅ Evaluated #{length(classifications)} classifications")

classifications
|> Enum.each(fn eval ->
  status = if(eval.passed, do: "✅", else: "❌")
  IO.puts("  #{status} #{eval.test_result.scenario_name}: " <>
          "expected='#{eval.test_result.expected_output}' " <>
          "actual='#{eval.test_result.actual_output}' " <>
          "(score: #{eval.scores.custom})")
end)

# ============================================================================
# Part 5: Multiple Evaluation Strategies
# ============================================================================

IO.puts("\n🎯 Part 5: Comparing Evaluation Strategies")
IO.puts("-" |> String.duplicate(70))

# Create a test result with text that has variations
text_result = %{
  scenario_id: "text_1",
  scenario_name: "Text generation",
  input: "Describe Elixir programming",
  expected_output: "Elixir is a dynamic functional programming language",
  actual_output: "Elixir is a functional programming language that is dynamic",
  latency_ms: 800,
  tokens_used: %{input: 5, output: 25},
  timestamp: DateTime.utc_now(),
  success: true,
  error: nil
}

# Compare different strategies
strategies = [
  {:exact_match, %{strategy: :exact_match, threshold: 1.0}},
  {:semantic_similarity, %{strategy: :semantic_similarity, threshold: 0.8}},
  {:semantic_similarity_low, %{strategy: :semantic_similarity, threshold: 0.5}}
]

IO.puts("Comparing strategies on same result:")
IO.puts("  Expected: #{text_result.expected_output}")
IO.puts("  Actual:   #{text_result.actual_output}")
IO.puts("")

strategies
|> Enum.each(fn {name, config} ->
  result = Llmgan.evaluate(text_result, config)
  passed = if(result.passed, do: "✅", else: "❌")

  score_display =
    case result.scores do
      %{overall: score} -> "overall=#{Float.round(score, 3)}"
      %{match: score} -> "match=#{score}"
      _ -> "N/A"
    end

  IO.puts("  #{passed} #{name} (#{score_display}): #{if(result.passed, do: "PASS", else: "FAIL")}")
end)

# ============================================================================
# Part 6: Report Generation
# ============================================================================

IO.puts("\n📊 Part 6: Comprehensive Report")
IO.puts("-" |> String.duplicate(70))

# Add all results to aggregator
Enum.each(sentiment_results, &Llmgan.ResultsAggregator.add_result/1)
Llmgan.ResultsAggregator.add_result(text_result)

report = Llmgan.generate_report()

IO.puts("""
Test Suite Summary:
  Total Scenarios: #{report.metrics.total_scenarios}
  Passed:          #{report.metrics.passed} (#{report.summary.pass_rate}%)
  Failed:          #{report.metrics.failed}
  Errors:          #{report.metrics.errors}

Performance Metrics:
  Average Latency: #{report.summary.avg_latency_ms}ms
  Total Tokens:    #{report.summary.total_tokens.input + report.summary.total_tokens.output}

Evaluator Performance:
""")

report.summary.evaluator_performance
|> Enum.each(fn perf ->
  IO.puts("  • #{perf.evaluator}: #{perf.count} tests, #{perf.pass_rate}% pass rate")
end)

# ============================================================================
# Part 7: Template Registration and Reuse
# ============================================================================

IO.puts("\n📝 Part 7: Template Registration and Reuse")
IO.puts("-" |> String.duplicate(70))

# Register a reusable template
template = %{
  id: "qa_template",
  name: "Question Answering Template",
  prompt_template: "Question: <%= question %>\nContext: <%= context %>",
  variables: ["question", "context"],
  metadata: %{domain: "qa", version: "1.0"}
}

:ok = Llmgan.register_template(template)
IO.puts("✅ Registered template: #{template.id}")

# Generate scenarios from registered template
{:ok, qa_scenarios} = Llmgan.generate_from_template("qa_template", %{
  question: "What is OTP?",
  context: "OTP is a set of Erlang libraries and design principles"
}, 3)

IO.puts("✅ Generated #{length(qa_scenarios)} scenarios from registered template")

qa_scenarios
|> Enum.each(fn s ->
  IO.puts("  • #{s.name}")
  IO.puts("    #{String.split(s.input, "\n") |> List.first()}")
end)

# ============================================================================
# Part 8: LLM-based Scenario Generation (Mock Demo)
# ============================================================================

IO.puts("\n🤖 Part 8: LLM-based Scenario Generation")
IO.puts("-" |> String.duplicate(70))
IO.puts("Using LLM to intelligently generate test scenarios with expected outputs...")

# For demo, show what the LLM would generate
mock_llm_response = ~S'''
{
  "scenarios": [
    {
      "name": "Simple arithmetic",
      "input": "What is 2 + 2?",
      "expected_output": "4"
    },
    {
      "name": "Negative numbers",
      "input": "What is -5 + 3?",
      "expected_output": "-2"
    },
    {
      "name": "Decimal arithmetic",
      "input": "What is 3.14 + 2.86?",
      "expected_output": "6"
    }
  ]
}
'''

# Simulate LLM generation result
llm_scenarios = [
  %{
    id: "llm_gen_1",
    name: "Simple arithmetic",
    input: "What is 2 + 2?",
    expected_output: "4",
    metadata: %{source: :llm_generated, domain: "math"},
    tags: ["llm_generated", "math"]
  },
  %{
    id: "llm_gen_2",
    name: "Negative numbers",
    input: "What is -5 + 3?",
    expected_output: "-2",
    metadata: %{source: :llm_generated, domain: "math"},
    tags: ["llm_generated", "math"]
  },
  %{
    id: "llm_gen_3",
    name: "Decimal arithmetic",
    input: "What is 3.14 + 2.86?",
    expected_output: "6",
    metadata: %{source: :llm_generated, domain: "math"},
    tags: ["llm_generated", "math"]
  }
]

IO.puts("✅ LLM generated #{length(llm_scenarios)} test scenarios")

llm_scenarios
|> Enum.each(fn s ->
  IO.puts("  • #{s.name}")
  IO.puts("    Input: #{s.input}")
  IO.puts("    Expected: #{s.expected_output}")
end)

# ============================================================================
# Cleanup
# ============================================================================

IO.puts("\n" <> "=" |> String.duplicate(70))
IO.puts("✨ Advanced example completed!")
IO.puts("=" |> String.duplicate(70))

Llmgan.reset()
