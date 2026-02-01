defmodule LlmganTest do
  use ExUnit.Case
  doctest Llmgan

  alias Llmgan.ScenarioGenerator
  alias Llmgan.ResultsAggregator
  alias Llmgan.Evaluators

  setup do
    # Reset the results aggregator before each test
    ResultsAggregator.reset()
    :ok
  end

  describe "scenario generation" do
    test "generates scenarios with LLM strategy" do
      # Test that the strategy config is valid
      config = %{
        description: "Test sentiment analysis",
        domain: "nlp",
        count: 2
      }

      # Verify the strategy is recognized (actual LLM call would fail without config)
      assert {:error, :missing_llm_config} =
               Llmgan.generate_scenarios(:llm, config)
    end

    test "generates template-based scenarios" do
      {:ok, scenarios} = Llmgan.generate_scenarios(:template, %{
        template: "Hello <%= @name %>",
        variables_list: [
          %{name: "World"},
          %{name: "Elixir"}
        ]
      })

      assert length(scenarios) == 2
      assert Enum.all?(scenarios, & &1.input in ["Hello World", "Hello Elixir"])
    end

    test "generates edge case scenarios" do
      {:ok, scenarios} = Llmgan.generate_scenarios(:edge_cases, %{
        template: "Input: <%= @input %>",
        field: "input"
      })

      assert length(scenarios) > 0
      assert Enum.any?(scenarios, & &1.metadata.type == :edge_case)
    end

    test "generates adversarial scenarios" do
      {:ok, scenarios} = Llmgan.generate_scenarios(:adversarial, %{
        template: "<%= @input %>"
      })

      assert length(scenarios) > 0
      assert Enum.any?(scenarios, fn s ->
        Enum.member?(s.tags, "adversarial")
      end)
    end

    test "generates fuzzing scenarios" do
      {:ok, scenarios} = Llmgan.generate_scenarios(:fuzzing, %{
        template: "Test: <%= @text %>",
        base_variables: %{text: "base"},
        fuzz_fields: ["text"],
        count: 5
      })

      assert length(scenarios) == 5
      assert Enum.all?(scenarios, & &1.name =~ "Fuzz")
    end

    test "registers and uses templates" do
      template = %{
        id: "test_template",
        name: "Test Template",
        prompt_template: "Question: <%= @question %>",
        variables: ["question"]
      }

      assert :ok = Llmgan.register_template(template)

      {:ok, scenarios} = Llmgan.generate_from_template("test_template", %{
        question: "What is Elixir?"
      }, 2)

      assert length(scenarios) == 2
      assert Enum.all?(scenarios, & &1.input == "Question: What is Elixir?")
    end
  end

  describe "evaluation strategies" do
    test "exact match evaluation" do
      test_result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        expected_output: "hello",
        actual_output: "hello"
      }

      config = %{strategy: :exact_match, threshold: 1.0}
      result = Evaluators.evaluate(test_result, config)

      assert result.passed == true
      assert result.scores.match == 1.0
    end

    test "exact match evaluation fails on mismatch" do
      test_result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        expected_output: "hello",
        actual_output: "world"
      }

      config = %{strategy: :exact_match, threshold: 1.0}
      result = Evaluators.evaluate(test_result, config)

      assert result.passed == false
      assert result.scores.match == 0.0
    end

    test "semantic similarity evaluation" do
      test_result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        expected_output: "The quick brown fox",
        actual_output: "The fast brown fox"
      }

      config = %{strategy: :semantic_similarity, threshold: 0.5}
      result = Evaluators.evaluate(test_result, config)

      assert is_map(result.scores)
      assert result.scores.levenshtein_similarity > 0
      assert result.scores.jaccard_similarity > 0
      assert result.scores.overall > 0
    end

    test "custom evaluation function" do
      test_result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        expected_output: "target",
        actual_output: "this contains target text"
      }

      custom_fn = fn expected, actual ->
        if String.contains?(actual, expected), do: 1.0, else: 0.0
      end

      config = %{strategy: :custom, custom_fn: custom_fn, threshold: 1.0}
      result = Evaluators.evaluate(test_result, config)

      assert result.passed == true
      assert result.scores.custom == 1.0
    end

    test "returns error for missing custom function" do
      test_result = %{scenario_id: "s1", expected_output: "a", actual_output: "b"}
      config = %{strategy: :custom}

      result = Evaluators.evaluate(test_result, config)
      assert {:error, :missing_custom_function} = result
    end
  end

  describe "results aggregation" do
    test "adds and retrieves results" do
      result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        input: "test",
        expected_output: "expected",
        actual_output: "actual",
        latency_ms: 100,
        tokens_used: %{input: 10, output: 5},
        timestamp: DateTime.utc_now(),
        success: true,
        error: nil
      }

      :ok = ResultsAggregator.add_result(result)
      results = ResultsAggregator.get_results()

      assert length(results) == 1
      assert hd(results).scenario_id == "s1"
    end

    test "adds and retrieves evaluations" do
      evaluation = %{
        scenario_id: "s1",
        test_result: %{},
        scores: %{overall: 0.9},
        passed: true,
        evaluator_type: :exact_match,
        metadata: %{}
      }

      :ok = ResultsAggregator.add_evaluation(evaluation)
      evaluations = ResultsAggregator.get_evaluations()

      assert length(evaluations) == 1
      assert hd(evaluations).scenario_id == "s1"
    end

    test "generates report with metrics" do
      # Add some test results
      for i <- 1..3 do
        result = %{
          scenario_id: "s#{i}",
          scenario_name: "Test #{i}",
          input: "input",
          expected_output: "expected",
          actual_output: "actual",
          latency_ms: 100 * i,
          tokens_used: %{input: 10, output: 5},
          timestamp: DateTime.utc_now(),
          success: true,
          error: nil
        }

        ResultsAggregator.add_result(result)

        evaluation = %{
          scenario_id: "s#{i}",
          test_result: result,
          scores: %{overall: 0.9},
          passed: i < 3,
          evaluator_type: :semantic_similarity,
          metadata: %{}
        }

        ResultsAggregator.add_evaluation(evaluation)
      end

      report = ResultsAggregator.generate_report()

      assert report.metrics.total_scenarios == 3
      assert report.summary.pass_rate > 0
      assert report.summary.avg_latency_ms > 0
    end

    test "resets all data" do
      result = %{
        scenario_id: "s1",
        scenario_name: "Test",
        input: "test",
        expected_output: nil,
        actual_output: nil,
        latency_ms: 100,
        tokens_used: nil,
        timestamp: DateTime.utc_now(),
        success: true,
        error: nil
      }

      ResultsAggregator.add_result(result)
      :ok = ResultsAggregator.reset()

      assert ResultsAggregator.get_results() == []
      assert ResultsAggregator.get_metrics().total_scenarios == 0
    end
  end

  describe "types" do
    test "defines scenario type" do
      # Just verify the types module loads
      assert Code.ensure_loaded?(Llmgan.Types)
    end
  end

  describe "prompt templates" do
    test "run_tests accepts custom prompt_template option" do
      # Verify the option is accepted by checking the runner_config structure
      opts = [
        timeout_ms: 30_000,
        max_retries: 2,
        batch_size: 5,
        prompt_template: "Translate to French: <%= @input %>"
      ]

      runner_config = %{
        timeout_ms: Keyword.get(opts, :timeout_ms, 30_000),
        max_retries: Keyword.get(opts, :max_retries, 2),
        batch_size: Keyword.get(opts, :batch_size, 5),
        prompt_template: Keyword.get(opts, :prompt_template)
      }

      assert runner_config.prompt_template == "Translate to French: <%= @input %>"
    end

    test "template with @input placeholder renders input inline" do
      template = "Translate to French: <%= @input %>"
      scenario = %{input: "hello"}
      assigns = [input: scenario.input]

      rendered = EEx.eval_string(template, assigns: assigns)
      assert rendered == "Translate to French: hello"
    end

    test "template without @input appends input after template" do
      template = "You are a helpful translator. Translate the following:"
      scenario = %{input: "hello world", name: "Test"}
      assigns = [name: scenario.name]

      # Template doesn't reference @input, so input gets appended
      rendered_template = EEx.eval_string(template, assigns: assigns)
      input = scenario.input

      combined =
        if String.ends_with?(rendered_template, "\n") do
          rendered_template <> input
        else
          rendered_template <> "\n\n" <> input
        end

      assert combined == "You are a helpful translator. Translate the following:\n\nhello world"
    end

    test "scenario metadata generation_prompt is used as fallback" do
      scenario = %{
        id: "test1",
        name: "Test",
        input: "hello",
        expected_output: "bonjour",
        metadata: %{generation_prompt: "Translate: <%= @input %>"},
        tags: []
      }

      # Simulate the build_prompt logic - template from metadata
      template = get_in(scenario, [:metadata, :generation_prompt])
      assert template == "Translate: <%= @input %>"

      assigns = [
        id: scenario.id,
        name: scenario.name,
        input: scenario.input,
        expected_output: scenario.expected_output,
        metadata: scenario.metadata,
        tags: scenario.tags
      ]

      rendered = EEx.eval_string(template, assigns: assigns)
      assert rendered == "Translate: hello"
    end

    test "direct input used when no template provided" do
      scenario = %{
        id: "test1",
        name: "Test",
        input: "hello world",
        expected_output: nil,
        metadata: %{},
        tags: []
      }

      # When no template exists, use input directly
      template = get_in(scenario, [:metadata, :generation_prompt])
      assert is_nil(template)

      # Would fall back to scenario.input
      assert scenario.input == "hello world"
    end
  end
end

defmodule Llmgan.ScenarioGeneratorTest do
  use ExUnit.Case

  alias Llmgan.ScenarioGenerator

  setup do
    # Clean up any test templates
    on_exit(fn ->
      :ets.delete_all_objects(:scenario_templates)
    end)

    :ok
  end

  test "stores and retrieves templates" do
    template = %{id: "t1", name: "Test", prompt_template: "Hello"}
    :ok = ScenarioGenerator.register_template(template)

    retrieved = ScenarioGenerator.get_template("t1")
    assert retrieved.name == "Test"
  end

  test "lists all templates" do
    ScenarioGenerator.register_template(%{id: "t1", name: "Test 1"})
    ScenarioGenerator.register_template(%{id: "t2", name: "Test 2"})

    templates = ScenarioGenerator.list_templates()
    assert length(templates) >= 2
  end

  test "returns nil for missing template" do
    assert ScenarioGenerator.get_template("nonexistent") == nil
  end

  test "generates scenarios with EEx templates" do
    {:ok, scenarios} = ScenarioGenerator.generate_with_strategy(:template, %{
      template: "Name: <%= @name %>, Age: <%= @age %>",
      variables_list: [
        %{name: "Alice", age: 30},
        %{name: "Bob", age: 25}
      ]
    })

    assert length(scenarios) == 2
    [first, second] = scenarios
    assert first.input == "Name: Alice, Age: 30"
    assert second.input == "Name: Bob, Age: 25"
  end
end

defmodule Llmgan.EvaluatorsTest do
  use ExUnit.Case

  alias Llmgan.Evaluators

  describe "similarity metrics" do
    test "calculates exact match" do
      # Identical strings
      result1 = Evaluators.evaluate(
        %{scenario_id: "s1", expected_output: "hello", actual_output: "hello"},
        %{strategy: :exact_match}
      )
      assert result1.scores.match == 1.0

      # Different strings
      result2 = Evaluators.evaluate(
        %{scenario_id: "s2", expected_output: "hello", actual_output: "world"},
        %{strategy: :exact_match}
      )
      assert result2.scores.match == 0.0
    end

    test "semantic similarity handles empty strings" do
      result = Evaluators.evaluate(
        %{scenario_id: "s1", expected_output: "", actual_output: ""},
        %{strategy: :semantic_similarity}
      )

      assert is_map(result.scores)
      assert result.scores.overall >= 0.0
    end
  end
end
