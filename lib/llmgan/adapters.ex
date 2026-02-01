defmodule Llmgan.Adapters do
  @moduledoc """
  Adapter behaviour and implementations for different LLM providers.

  Provides a unified interface for working with various LLM APIs:
  - OpenAI (and compatible APIs)
  - Anthropic Claude
  - Google AI (Gemini)
  - Local models via Ollama

  ## Example

      adapter = Llmgan.Adapters.get(:openai)
      {:ok, response} = adapter.call(adapter_config, messages)
  """

  @doc """
  Callback for making a call to the LLM API.
  """
  @callback call(config :: map(), messages :: list()) ::
              {:ok, response :: String.t(), metadata :: map()} | {:error, reason :: term()}

  @doc """
  Gets the adapter module for the specified provider.
  """
  def get(provider) do
    case provider do
      :openai -> Llmgan.Adapters.OpenAI
      :anthropic -> Llmgan.Adapters.Anthropic
      :google -> Llmgan.Adapters.GoogleAI
      :ollama -> Llmgan.Adapters.Ollama
      _ -> raise ArgumentError, "Unknown provider: #{inspect(provider)}"
    end
  end

  @doc """
  Lists available providers.
  """
  def available_providers do
    [:openai, :anthropic, :google, :ollama]
  end
end

defmodule Llmgan.Adapters.OpenAI do
  @moduledoc """
  Adapter for OpenAI API.
  """

  @behaviour Llmgan.Adapters

  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @impl true
  def call(config, messages) do
    try do
      llm =
        ChatOpenAI.new!(%{
          model: config.model,
          temperature: Map.get(config, :temperature, 0.7),
          max_tokens: Map.get(config, :max_tokens),
          api_key: config.api_key,
          endpoint: Map.get(config, :endpoint),
          stream: false
        })

      chain =
        LLMChain.new!(%{llm: llm})
        |> add_messages(messages)

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)
          content = extract_content(last_message)

          metadata = %{
            tokens_used: get_in(last_message, [Access.key(:metadata), :usage]),
            model: config.model
          }

          {:ok, content, metadata}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp add_messages(chain, messages) do
    Enum.reduce(messages, chain, fn
      %{role: :system, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_system!(content))

      %{role: :user, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_user!(content))

      %{role: :assistant, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_assistant!(content))

      text, acc when is_binary(text) ->
        LLMChain.add_message(acc, Message.new_user!(text))

      _, acc ->
        acc
    end)
  end

  defp extract_content(message) do
    case message do
      %{content: content} when is_binary(content) -> content
      %{content: parts} when is_list(parts) -> Enum.map_join(parts, "\n", & &1.content)
      _ -> ""
    end
  end
end

defmodule Llmgan.Adapters.Anthropic do
  @moduledoc """
  Adapter for Anthropic Claude API.
  """

  @behaviour Llmgan.Adapters

  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @impl true
  def call(config, messages) do
    try do
      llm =
        ChatAnthropic.new!(%{
          model: config.model,
          temperature: Map.get(config, :temperature, 0.7),
          max_tokens: Map.get(config, :max_tokens, 1024),
          api_key: config.api_key
        })

      chain =
        LLMChain.new!(%{llm: llm})
        |> add_messages(messages)

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)
          content = extract_content(last_message)

          metadata = %{
            tokens_used: get_in(last_message, [Access.key(:metadata), :usage]),
            model: config.model
          }

          {:ok, content, metadata}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp add_messages(chain, messages) do
    Enum.reduce(messages, chain, fn
      %{role: :system, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_system!(content))

      %{role: :user, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_user!(content))

      %{role: :assistant, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_assistant!(content))

      text, acc when is_binary(text) ->
        LLMChain.add_message(acc, Message.new_user!(text))

      _, acc ->
        acc
    end)
  end

  defp extract_content(message) do
    case message do
      %{content: content} when is_binary(content) -> content
      %{content: parts} when is_list(parts) -> Enum.map_join(parts, "\n", & &1.content)
      _ -> ""
    end
  end
end

defmodule Llmgan.Adapters.GoogleAI do
  @moduledoc """
  Adapter for Google AI (Gemini) API.
  """

  @behaviour Llmgan.Adapters

  alias LangChain.ChatModels.ChatGoogleAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @impl true
  def call(config, messages) do
    try do
      llm =
        ChatGoogleAI.new!(%{
          model: config.model,
          temperature: Map.get(config, :temperature, 0.7),
          max_tokens: Map.get(config, :max_tokens),
          api_key: config.api_key
        })

      chain =
        LLMChain.new!(%{llm: llm})
        |> add_messages(messages)

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)
          content = extract_content(last_message)

          metadata = %{
            tokens_used: get_in(last_message, [Access.key(:metadata), :usage]),
            model: config.model
          }

          {:ok, content, metadata}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp add_messages(chain, messages) do
    Enum.reduce(messages, chain, fn
      %{role: :system, content: content}, acc ->
        # Google AI may handle system messages differently
        LLMChain.add_message(acc, Message.new_user!("System: #{content}"))

      %{role: :user, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_user!(content))

      %{role: :assistant, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_assistant!(content))

      text, acc when is_binary(text) ->
        LLMChain.add_message(acc, Message.new_user!(text))

      _, acc ->
        acc
    end)
  end

  defp extract_content(message) do
    case message do
      %{content: content} when is_binary(content) -> content
      %{content: parts} when is_list(parts) -> Enum.map_join(parts, "\n", & &1.content)
      _ -> ""
    end
  end
end

defmodule Llmgan.Adapters.Ollama do
  @moduledoc """
  Adapter for local models via Ollama.
  """

  @behaviour Llmgan.Adapters

  alias LangChain.ChatModels.ChatOllamaAI
  alias LangChain.Chains.LLMChain
  alias LangChain.Message

  @impl true
  def call(config, messages) do
    try do
      llm =
        ChatOllamaAI.new!(%{
          model: config.model,
          temperature: Map.get(config, :temperature, 0.7),
          endpoint: Map.get(config, :endpoint, "http://localhost:11434")
        })

      chain =
        LLMChain.new!(%{llm: llm})
        |> add_messages(messages)

      case LLMChain.run(chain) do
        {:ok, updated_chain} ->
          last_message = List.last(updated_chain.messages)
          content = extract_content(last_message)

          metadata = %{
            model: config.model
          }

          {:ok, content, metadata}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp add_messages(chain, messages) do
    Enum.reduce(messages, chain, fn
      %{role: :system, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_system!(content))

      %{role: :user, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_user!(content))

      %{role: :assistant, content: content}, acc ->
        LLMChain.add_message(acc, Message.new_assistant!(content))

      text, acc when is_binary(text) ->
        LLMChain.add_message(acc, Message.new_user!(text))

      _, acc ->
        acc
    end)
  end

  defp extract_content(message) do
    case message do
      %{content: content} when is_binary(content) -> content
      %{content: parts} when is_list(parts) -> Enum.map_join(parts, "\n", & &1.content)
      _ -> ""
    end
  end
end
