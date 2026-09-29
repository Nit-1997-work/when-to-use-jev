# when-to-use-jev
Benchmarking JEV against an LLM for intent classification on a public dataset: accuracy, latency, cost, and when each one is the right choice.

All LLM calls go through a LiteLLM-based gateway compatible with OpenAI's Chat Completions API; there are no direct Gemini API calls.
See [docs/experiment-design.md](docs/experiment-design.md) for the experiment design.
