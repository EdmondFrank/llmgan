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
end
