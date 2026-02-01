defmodule Llmgan.MixProject do
  use Mix.Project

  def project do
    [
      app: :llmgan,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Llmgan.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:langchain, "~> 0.3.0"},
      {:poolboy, "~> 1.5"},
      {:jason, "~> 1.4"},
      {:req, "~> 0.5"},
      {:telemetry, "~> 1.0"}
    ]
  end
end
