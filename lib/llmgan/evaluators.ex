defmodule Llmgan.Evaluators do
  @moduledoc """
  Evaluation strategies for LLM test results.

  Supports:
  - Exact match
  - Semantic similarity (using embeddings or string similarity)
  - LLM-as-judge (using another LLM to evaluate)
  - Custom evaluation functions
  """

  require Logger

  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @doc """
  Evaluates a test result using the configured strategy.
  """
  def evaluate(test_result, evaluation_config) do
    strategy = Map.get(evaluation_config, :strategy, :exact_match)

    case strategy do
      :exact_match ->
        evaluate_exact_match(test_result, evaluation_config)

      :semantic_similarity ->
        evaluate_semantic_similarity(test_result, evaluation_config)

      :llm_judge ->
        evaluate_llm_judge(test_result, evaluation_config)

      :custom ->
        evaluate_custom(test_result, evaluation_config)

      :json_schema ->
        evaluate_json_schema(test_result, evaluation_config)

      :json_field_match ->
        evaluate_json_field_match(test_result, evaluation_config)

      _ ->
        {:error, {:unknown_strategy, strategy}}
    end
  end

  @doc """
  Exact match evaluation - compares expected and actual outputs exactly.
  """
  def evaluate_exact_match(test_result, evaluation_config) do
    expected = normalize_output(test_result.expected_output)
    actual = normalize_output(test_result.actual_output)

    passed = expected == actual

    scores = %{
      match: if(passed, do: 1.0, else: 0.0),
      similarity: if(passed, do: 1.0, else: 0.0)
    }

    threshold = Map.get(evaluation_config, :threshold, 1.0)

    %{
      scenario_id: test_result.scenario_id,
      test_result: test_result,
      scores: scores,
      passed: scores.match >= threshold,
      evaluator_type: :exact_match,
      metadata: %{
        expected_length: String.length(to_string(expected)),
        actual_length: String.length(to_string(actual))
      }
    }
  end

  @doc """
  Semantic similarity evaluation using string similarity metrics.
  """
  def evaluate_semantic_similarity(test_result, evaluation_config) do
    expected = normalize_output(test_result.expected_output)
    actual = normalize_output(test_result.actual_output)

    # Calculate multiple similarity metrics
    scores = %{
      levenshtein_similarity: levenshtein_similarity(expected, actual),
      jaccard_similarity: jaccard_similarity(expected, actual),
      cosine_similarity: cosine_similarity(expected, actual)
    }

    # Weighted average of similarity scores
    overall_score =
      scores.levenshtein_similarity * 0.4 +
        scores.jaccard_similarity * 0.3 +
        scores.cosine_similarity * 0.3

    scores = Map.put(scores, :overall, overall_score)

    threshold = Map.get(evaluation_config, :threshold, 0.8)

    %{
      scenario_id: test_result.scenario_id,
      test_result: test_result,
      scores: scores,
      passed: overall_score >= threshold,
      evaluator_type: :semantic_similarity,
      metadata: %{
        expected_length: String.length(to_string(expected)),
        actual_length: String.length(to_string(actual))
      }
    }
  end

  @doc """
  LLM-as-judge evaluation using another LLM to evaluate the response quality.
  """
  def evaluate_llm_judge(test_result, evaluation_config) do
    llm_config = Map.get(evaluation_config, :llm_config)

    if is_nil(llm_config) do
      {:error, :missing_llm_config}
    else
      prompt = build_judge_prompt(test_result)

      case call_judge_llm(llm_config, prompt) do
        {:ok, evaluation} ->
          scores = parse_judge_evaluation(evaluation)

          threshold = Map.get(evaluation_config, :threshold, 0.7)

          %{
            scenario_id: test_result.scenario_id,
            test_result: test_result,
            scores: scores,
            passed: scores.overall >= threshold,
            evaluator_type: :llm_judge,
            metadata: %{
              raw_evaluation: evaluation,
              judge_model: llm_config.model
            }
          }

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @doc """
  JSON Schema validation - validates that actual output conforms to expected JSON schema.
  """
  def evaluate_json_schema(test_result, evaluation_config) do
    schema = Map.get(evaluation_config, :json_schema)

    if is_nil(schema) do
      {:error, :missing_json_schema}
    else
      actual = test_result.actual_output

      case parse_json(actual) do
        {:ok, parsed} ->
          validation_result = validate_schema(parsed, schema)
          passed = validation_result.valid

          scores = %{
            schema_valid: if(passed, do: 1.0, else: 0.0),
            field_match_rate: validation_result.field_match_rate,
            required_fields_present: validation_result.required_present,
            overall: validation_result.field_match_rate
          }

          threshold = Map.get(evaluation_config, :threshold, 1.0)

          %{
            scenario_id: test_result.scenario_id,
            test_result: test_result,
            scores: scores,
            passed: passed and scores.overall >= threshold,
            evaluator_type: :json_schema,
            metadata: %{
              schema: schema,
              validation_errors: validation_result.errors,
              matched_fields: validation_result.matched_fields,
              total_fields: validation_result.total_fields
            }
          }

        {:error, reason} ->
          %{
            scenario_id: test_result.scenario_id,
            test_result: test_result,
            scores: %{schema_valid: 0.0, field_match_rate: 0.0},
            passed: false,
            evaluator_type: :json_schema,
            metadata: %{
              error: "Invalid JSON: #{reason}",
              raw_output: actual
            }
          }
      end
    end
  end

  @doc """
  JSON Field Match evaluation - validates specific fields in JSON output.
  """
  def evaluate_json_field_match(test_result, evaluation_config) do
    matchers = Map.get(evaluation_config, :field_matchers, [])

    actual = test_result.actual_output

    case parse_json(actual) do
      {:ok, parsed} ->
        results =
          Enum.map(matchers, fn matcher ->
            path = matcher.path
            expected = matcher.expected
            match_type = Map.get(matcher, :match_type, :exact)

            actual_value = get_value_at_path(parsed, path)
            match_result = match_value(actual_value, expected, match_type)

            %{
              path: path,
              expected: expected,
              actual: actual_value,
              matched: match_result.matched,
              match_type: match_type,
              error: match_result.error
            }
          end)

        matched_count = Enum.count(results, & &1.matched)
        total_count = length(results)
        match_rate = if total_count > 0, do: matched_count / total_count, else: 1.0

        scores = %{
          field_match_rate: match_rate,
          fields_matched: matched_count,
          fields_total: total_count
        }

        threshold = Map.get(evaluation_config, :threshold, 1.0)

        %{
          scenario_id: test_result.scenario_id,
          test_result: test_result,
          scores: scores,
          passed: match_rate >= threshold,
          evaluator_type: :json_field_match,
          metadata: %{
            field_results: results,
            unmatched_fields: Enum.reject(results, & &1.matched)
          }
        }

      {:error, reason} ->
        %{
          scenario_id: test_result.scenario_id,
          test_result: test_result,
          scores: %{field_match_rate: 0.0},
          passed: false,
          evaluator_type: :json_field_match,
          metadata: %{
            error: "Invalid JSON: #{reason}",
            raw_output: actual
          }
        }
    end
  end

  @doc """
  Custom evaluation using a user-provided function.
  """
  def evaluate_custom(test_result, evaluation_config) do
    custom_fn = Map.get(evaluation_config, :custom_fn)

    if is_nil(custom_fn) or not is_function(custom_fn) do
      {:error, :missing_custom_function}
    else
      try do
        score = custom_fn.(test_result.expected_output, test_result.actual_output)

        threshold = Map.get(evaluation_config, :threshold, 0.5)

        %{
          scenario_id: test_result.scenario_id,
          test_result: test_result,
          scores: %{custom: score},
          passed: score >= threshold,
          evaluator_type: :custom,
          metadata: %{}
        }
      rescue
        e ->
          {:error, {:custom_evaluation_failed, Exception.message(e)}}
      end
    end
  end

  # Private Helper Functions

  defp normalize_output(nil), do: ""
  defp normalize_output(output) when is_binary(output), do: String.trim(output)
  defp normalize_output(output), do: Jason.encode!(output)

  # Levenshtein distance similarity (0.0 to 1.0)
  defp levenshtein_similarity(str1, str2) do
    distance = levenshtein_distance(str1, str2)
    max_len = max(String.length(str1), String.length(str2))

    if max_len == 0 do
      1.0
    else
      1.0 - distance / max_len
    end
  end

  defp levenshtein_distance(str1, str2) do
    len1 = String.length(str1)
    len2 = String.length(str2)

    # Handle edge cases
    cond do
      len1 == 0 ->
        len2

      len2 == 0 ->
        len1

      true ->
        # Simple dynamic programming implementation
        matrix =
          for i <- 0..len1 do
            for j <- 0..len2 do
              cond do
                i == 0 -> j
                j == 0 -> i
                true -> 0
              end
            end
          end

        matrix =
          for i <- 1..len1, j <- 1..len2, reduce: matrix do
            acc ->
              char1 = String.at(str1, i - 1)
              char2 = String.at(str2, j - 1)

              cost = if char1 == char2, do: 0, else: 1

              deletion = Enum.at(Enum.at(acc, i - 1), j) + 1
              insertion = Enum.at(Enum.at(acc, i), j - 1) + 1
              substitution = Enum.at(Enum.at(acc, i - 1), j - 1) + cost

              new_val = min(min(deletion, insertion), substitution)

              List.replace_at(
                acc,
                i,
                List.replace_at(Enum.at(acc, i), j, new_val)
              )
          end

        List.last(List.last(matrix))
    end
  end

  # Jaccard similarity for word sets
  defp jaccard_similarity(str1, str2) do
    words1 = str1 |> String.downcase() |> String.split() |> MapSet.new()
    words2 = str2 |> String.downcase() |> String.split() |> MapSet.new()

    intersection = MapSet.size(MapSet.intersection(words1, words2))
    union = MapSet.size(MapSet.union(words1, words2))

    if union == 0, do: 1.0, else: intersection / union
  end

  # Cosine similarity for character bigrams
  defp cosine_similarity(str1, str2) do
    bigrams1 = extract_bigrams(str1)
    bigrams2 = extract_bigrams(str2)

    all_bigrams = MapSet.union(MapSet.new(bigrams1), MapSet.new(bigrams2))

    vec1 = Enum.map(all_bigrams, &count_occurrences(&1, bigrams1))
    vec2 = Enum.map(all_bigrams, &count_occurrences(&1, bigrams2))

    dot_product = Enum.zip_with(vec1, vec2, &(&1 * &2)) |> Enum.sum()
    mag1 = :math.sqrt(Enum.sum(Enum.map(vec1, &(&1 * &1))))
    mag2 = :math.sqrt(Enum.sum(Enum.map(vec2, &(&1 * &1))))

    if mag1 == 0 or mag2 == 0, do: 0.0, else: dot_product / (mag1 * mag2)
  end

  defp extract_bigrams(str) do
    chars = String.graphemes(str)

    chars
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(fn [a, b] -> a <> b end)
  end

  defp count_occurrences(item, list) do
    Enum.count(list, &(&1 == item))
  end

  defp build_judge_prompt(test_result) do
    expected = test_result.expected_output || "No expected output provided"
    actual = test_result.actual_output || "No actual output"
    input = test_result.input

    """
    You are an expert evaluator of AI responses. Please evaluate the following response:

    INPUT:
    #{format_for_judge(input)}

    EXPECTED OUTPUT:
    #{format_for_judge(expected)}

    ACTUAL OUTPUT:
    #{format_for_judge(actual)}

    Evaluate based on:
    1. Accuracy (0-1): How accurate is the response compared to expected?
    2. Completeness (0-1): Does it cover all aspects?
    3. Relevance (0-1): Is it relevant to the input?
    4. Overall (0-1): Overall quality score

    Provide your evaluation in this JSON format:
    {
      "accuracy": <score>,
      "completeness": <score>,
      "relevance": <score>,
      "overall": <score>,
      "reasoning": "<brief explanation>"
    }
    """
  end

  defp format_for_judge(content) when is_binary(content), do: content
  defp format_for_judge(content), do: inspect(content)

  defp call_judge_llm(llm_config, prompt) do
    try do
      llm = create_llm_model(llm_config)

      chain =
        LLMChain.new!(%{llm: llm})
        |> LLMChain.add_message(Message.new_user!(prompt))

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)
          content = extract_content(last_message)
          {:ok, content}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp parse_judge_evaluation(evaluation_text) do
    # Try to extract JSON from the response
    json_pattern = ~r/\{[^}]*\}/s

    case Regex.scan(json_pattern, evaluation_text) do
      [[json_str]] ->
        case Jason.decode(json_str) do
          {:ok, parsed} ->
            %{
              accuracy: Map.get(parsed, "accuracy", 0.0),
              completeness: Map.get(parsed, "completeness", 0.0),
              relevance: Map.get(parsed, "relevance", 0.0),
              overall: Map.get(parsed, "overall", 0.0),
              reasoning: Map.get(parsed, "reasoning", "")
            }

          {:error, _} ->
            parse_text_evaluation(evaluation_text)
        end

      _ ->
        parse_text_evaluation(evaluation_text)
    end
  end

  defp parse_text_evaluation(text) do
    # Fallback: try to extract scores from text
    overall_score = extract_score(text, ~r/overall[:\s]+(\d+\.?\d*)/i)

    %{
      accuracy: extract_score(text, ~r/accuracy[:\s]+(\d+\.?\d*)/i),
      completeness: extract_score(text, ~r/completeness[:\s]+(\d+\.?\d*)/i),
      relevance: extract_score(text, ~r/relevance[:\s]+(\d+\.?\d*)/i),
      overall: overall_score,
      reasoning: text
    }
  end

  defp extract_score(text, pattern) do
    case Regex.run(pattern, text) do
      [_, score_str] ->
        case Float.parse(score_str) do
          {score, _} -> min(max(score, 0.0), 1.0)
          :error -> 0.5
        end

      _ ->
        0.5
    end
  end

  defp create_llm_model(llm_config) do
    case llm_config.provider do
      :openai ->
        alias LangChain.ChatModels.ChatOpenAI

        ChatOpenAI.new!(%{
          model: llm_config.model,
          temperature: 0.1,
          max_tokens: 1000,
          api_key: llm_config.api_key,
          endpoint: Map.get(llm_config, :endpoint, "https://api.openai.com/v1")
        })

      :anthropic ->
        alias LangChain.ChatModels.ChatAnthropic

        ChatAnthropic.new!(%{
          model: llm_config.model,
          temperature: 0.1,
          max_tokens: 1000,
          api_key: llm_config.api_key,
          endpoint: Map.get(llm_config, :endpoint, "https://api.anthropic.com")
        })

      _ ->
        raise ArgumentError, "Unsupported LLM provider for judge"
    end
  end

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

  # JSON Helper Functions

  defp parse_json(value) when is_map(value), do: {:ok, value}

  defp parse_json(value) when is_binary(value) do
    # Try to extract JSON from markdown code blocks or raw JSON
    json_str = extract_json_from_text(value)

    case Jason.decode(json_str) do
      {:ok, parsed} ->
        {:ok, parsed}

      {:error, _} ->
        # Try to parse as Elixir map string
        case Code.string_to_quoted(json_str) do
          {:ok, quoted} when is_list(quoted) ->
            try do
              {:ok, Enum.into(quoted, %{})}
            rescue
              _ -> {:error, "Cannot parse JSON"}
            end

          {:ok, quoted} when is_map(quoted) ->
            try do
              {:ok, Macro.escape(quoted) |> Code.eval_quoted() |> elem(0)}
            rescue
              _ -> {:error, "Cannot parse JSON"}
            end

          _ ->
            {:error, "Invalid JSON format"}
        end
    end
  end

  defp parse_json(_), do: {:error, "Unsupported type"}

  defp extract_json_from_text(text) do
    # Try to extract JSON from markdown code blocks
    case Regex.run(~r/```json\s*\n(.*?)\n```/s, text) do
      [_, json] ->
        json

      _ ->
        # Try to extract from single backticks
        case Regex.run(~r/`(.*?)`/s, text) do
          [_, json] -> json
          _ -> text
        end
    end
  end

  defp validate_schema(data, schema) do
    required = Map.get(schema, "required", [])
    properties = Map.get(schema, "properties", %{})

    # Check required fields
    missing_required = Enum.reject(required, &Map.has_key?(data, &1))
    required_present = length(missing_required) == 0

    # Validate each property
    field_results =
      Enum.map(properties, fn {field, field_schema} ->
        value = Map.get(data, field)
        validate_field(field, value, field_schema)
      end)

    matched_fields = Enum.count(field_results, & &1.valid)
    total_fields = length(field_results)
    field_match_rate = if total_fields > 0, do: matched_fields / total_fields, else: 1.0

    errors =
      if(required_present,
        do: [],
        else: ["Missing required fields: #{Enum.join(missing_required, ", ")}"]
      ) ++
        Enum.flat_map(field_results, & &1.errors)

    %{
      valid: required_present and field_match_rate == 1.0,
      required_present: required_present,
      field_match_rate: field_match_rate,
      matched_fields: matched_fields,
      total_fields: total_fields,
      errors: errors
    }
  end

  defp validate_field(field, value, schema) do
    expected_type = Map.get(schema, "type")

    type_valid =
      case expected_type do
        "string" -> is_binary(value)
        "integer" -> is_integer(value)
        "number" -> is_number(value)
        "boolean" -> is_boolean(value)
        "array" -> is_list(value)
        "object" -> is_map(value)
        nil -> true
        _ -> true
      end

    %{
      field: field,
      valid: type_valid,
      errors:
        if(type_valid,
          do: [],
          else: ["Field '#{field}': expected #{expected_type}, got #{typeof(value)}"]
        )
    }
  end

  defp typeof(value) when is_binary(value), do: "string"
  defp typeof(value) when is_integer(value), do: "integer"
  defp typeof(value) when is_number(value), do: "number"
  defp typeof(value) when is_boolean(value), do: "boolean"
  defp typeof(value) when is_list(value), do: "array"
  defp typeof(value) when is_map(value), do: "object"
  defp typeof(_), do: "unknown"

  defp get_value_at_path(data, path) when is_binary(path) do
    parts = String.split(path, ".")
    get_value_at_path(data, parts)
  end

  defp get_value_at_path(data, []), do: data

  defp get_value_at_path(data, [key | rest]) do
    value = Map.get(data, key) || Map.get(data, String.to_atom(key))
    get_value_at_path(value, rest)
  end

  defp get_value_at_path(_nil, _), do: nil

  defp match_value(actual, expected, :exact) do
    matched = actual == expected

    %{
      matched: matched,
      error: if(matched, do: nil, else: "Expected #{inspect(expected)}, got #{inspect(actual)}")
    }
  end

  defp match_value(actual, expected, :contains) when is_binary(actual) and is_binary(expected) do
    matched = String.contains?(actual, expected)

    %{
      matched: matched,
      error: if(matched, do: nil, else: "Expected to contain '#{expected}', got '#{actual}'")
    }
  end

  defp match_value(actual, expected, :contains) when is_list(actual) do
    matched = Enum.member?(actual, expected)

    %{
      matched: matched,
      error: if(matched, do: nil, else: "Expected list to contain #{inspect(expected)}")
    }
  end

  defp match_value(actual, expected, :regex) when is_binary(actual) and is_binary(expected) do
    case Regex.compile(expected) do
      {:ok, regex} ->
        matched = Regex.match?(regex, actual)

        %{
          matched: matched,
          error: if(matched, do: nil, else: "Expected to match pattern '#{expected}'")
        }

      {:error, reason} ->
        %{matched: false, error: "Invalid regex pattern: #{reason}"}
    end
  end

  defp match_value(actual, _expected, :type) do
    matched = actual != nil
    %{matched: matched, error: if(matched, do: nil, else: "Value is nil")}
  end

  defp match_value(actual, expected, _unknown_type) do
    match_value(actual, expected, :exact)
  end
end
