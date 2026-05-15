# 🚀 OpenAI Technical Bootcamp - Notes & Resources

> 📘 Practical notes, examples, and official resources covering
> evaluation systems, prompt optimization, agentic architectures,
> multimodality, Codex workflows, and production best practices.

------------------------------------------------------------------------

## 🧪 1. Evaluation Systems (Evals)

Evals act as **unit tests for LLM applications**, validating accuracy,
reliability, and regressions across prompts, models, and pipelines.

### 🎯 Core Mental Model

**🧪 Evals = Unit tests for AI systems**

> In traditional software we write unit tests.  
> Here we do the same for GenAI pipelines - prompts, agents, RAG systems, and multimodal workflows.

- **Dataset** = real-world scenarios  
- **Model output** = what the AI generates  
- **Grader** = checks correctness/quality  
- **Scores** = track performance over time
- **Goal** → reliability + continuous improvement  

### ✅ Core Capabilities

-   📊 Structured datasets and schemas\
-   🤖 Automated grading (rule-based + model-based)\
-   🔁 Bulk experiments across prompts/models\
-   🧯 CI regression guardrails\
-   📈 Quality metrics tied to business impact (cost, errors, QA effort)

**The eval loop**
- **Dataset / test cases** → **Prompt / pipeline** → **Model outputs** → **Graders** → **Pass/Fail + scores** → **Iterate**

**Where evals fit**
- Prompt changes, model upgrades, tool/agent changes, RAG changes, multimodal extraction changes.

---

### 🧺 Quick Start: Test Data & Datasets (programmatic)

You need a **representative test set** to evaluate prompts and pipelines. “Dataset” here simply means a collection of test items (CSV/JSONL) + optional ground truth.

**How to structure test data**
- Prefer **JSONL** where each line is: `{ "item": { ...fields... } }`
- Include:
  - **Inputs** (e.g., `ticket_text`, `receipt_image_path`)
  - **Ground truth** fields where possible (e.g., `correct_label`, `correct_total`)

**Where it comes from**
- Production logs (after de-identification)
- Domain expert-labeled samples
- Synthetic edge cases (generated, then reviewed)

**Minimal workflow**
1. Start with 50–200 cases (happy path + edge cases).
2. Add ground truth for what “correct” means (labels, extracted fields, etc.).
3. Expand continuously as you discover failures.

**When you don’t have ground truth**
- Use model graders (LLM-as-judge) + human spot-checking, then gradually backfill labels.

---

### 🧰 Graders (automated scoring types)

Common grader types (use one or combine many):

- **String check**: exact/contains match against a reference column
- **Text similarity**: semantic closeness (useful when exact match isn’t required)
- **Score model**: LLM returns a numeric score (good for subjective criteria)
- **Label model**: LLM assigns a label like “correct/incorrect”, “concise/verbose”
- **Python**: custom logic (e.g., enforce word count, schema validation)

**Templating**
- `{{ item.* }}` → fields from dataset/test item (ground truth, inputs)
- `{{ sample.output_text }}` → model output content
- For tool-call grading, use `{{ sample.output_tools }}` and consider combining graders.

**Design tips**
- Prefer **smooth scores** (not only binary) when you plan to optimize.
- Guard against “grader hacking” and biases (verbosity/position/style bias).
- Calibrate automated graders with occasional human review.

---

### 🧱 Evals API (programmable, scalable, CI-friendly)

Evals are the scalable version of the workflow:
1. **Describe the task as an eval**
   - Define `data_source_config` (JSON schema for items)
   - Define `testing_criteria` (your graders)
2. **Run evals** on test inputs (prompt + data)
3. **Analyze results** via dashboard report URL or API and iterate

### 🧩 How an Eval is Structured (API mental model)

Evals are created by defining:

1) Dataset schema + items  
2) One or more graders (testing_criteria)  
3) Runs that either generate outputs or score precomputed outputs  

✅ Two common patterns

Pattern A (live generation)
input only (+ optional ground truth)  
model generates output  
graders judge it  

Pattern B (offline evaluation)
input + output + ground truth  
no generation  
graders only compare fields

**Big takeaway**
> GenAI systems can be continuously tested, monitored, and improved just like software - evals make reliability measurable instead of subjective.
---

### 📌 Example Use Case

#### 🧾 Receipt Extraction + Audit Pipeline

- Image → structured JSON extraction → graders validate totals/merchant/items (including missed items) → decide “processed vs needs audit” → tie eval metrics to business cost/risk.
-   Multimodal extraction (image → structured JSON)\
-   Ground truth vs predicted comparisons\
-   String checks, similarity metrics, model graders\
-   Missed-item detection\
-   Pass/fail thresholds tied to audit decisions

### 🔗 Official Docs

-   https://platform.openai.com/docs/guides/evaluation-getting-started\
-   https://platform.openai.com/docs/guides/evals\
-   https://platform.openai.com/docs/guides/graders\
-   https://platform.openai.com/docs/guides/evaluation-best-practices\
-   https://platform.openai.com/docs/guides/model-optimization

------------------------------------------------------------------------

## 🧠 2. Prompt Optimization & LLM Accuracy

Prompt optimization is a **systematic, eval-driven process** - not trial-and-error.

### 🎯 Core Mental Model: Accuracy vs Behavior

LLM improvement happens across two main axes:

**Context optimization (accuracy)**
- Missing domain knowledge
- Outdated info
- Proprietary data
→ solved with **better context + RAG**

**LLM optimization (consistency & behavior)**
- Formatting issues
- Style drift
- Reasoning inconsistency
→ solved with **prompting + fine-tuning**

**Baseline always starts with prompt engineering + evals**

Evaluate → diagnose failure → pull correct lever → re-evaluate

---

### 🪜 Typical Optimization Ladder

1. Start with clear prompt + eval set  
2. Add instructions & few-shot examples (behavior tuning)  
3. Add RAG for dynamic context (accuracy tuning)  
4. Fine-tune when consistency still fails  
5. Stack RAG + fine-tuning if needed  

**Key rule:** squeeze maximum performance from prompts before RAG or fine-tuning.

---

### ✍️ Prompt Engineering (API patterns)

Best practices that actually scale:

- Clear role + task instructions  
- Break complex tasks into subtasks  
- Few-shot examples for consistency  
- Structured formatting (Markdown/XML)  
- Explicit constraints on outputs  
- Pin model versions + measure with evals  

---

### 📈 Evaluation-driven prompting

Prompt work should always end with:

- Test dataset (real scenarios)  
- Ground truth or model graders  
- Automated scoring  

Common techniques:
- Similarity metrics 
- LLM-as-judge graders  
- Regression tracking over time  

If you can’t measure it - you can’t improve it.

---

### 📚 Retrieval-Augmented Generation (RAG)

RAG = dynamically inject domain context before generation.

Used when:
- Knowledge not in model
- Private/internal data
- Fresh content

Two failure zones to evaluate:

| Area | Failure | Fix |
|-----|--------|-----|
| Retrieval | Wrong or noisy context | Tune search, chunking, embeddings |
| LLM | Misuses correct context | Improve prompt or fine-tune |

RAG improves **accuracy**, not behavior.

---

### 🎯 Fine-tuning

Used when the model must learn consistent task behavior.

Goals:
- Higher task accuracy
- Fewer prompt tokens
- Strong formatting reliability

Best practices:
- Start with 50–100 high-quality examples
- Use production-like data
- Keep eval holdout set
- Stack with RAG when needed

Fine-tuning fixes **learned memory problems** (When a model consistently repeats something incorrectly, retraining it on corrected examples can permanently change that behavior.)

---

### 🧠 Reasoning Models vs GPT Models

**Reasoning models**
- Complex planning
- Multistep decisions
- Ambiguous problems
- Agent orchestration
- Deep document reasoning

**GPT models**
- Fast execution
- Well-defined tasks
- High throughput
- Lower cost

**Real systems use both:**

Planner (reasoning) → Executor (GPT)

---

### ⚙️ Reasoning via Responses API

Key controls:

- `reasoning.effort`: low | medium | high  
- Reserve token space for reasoning  
- Track reasoning token cost  
- Use summaries when needed  

Higher effort = better multistep intelligence.

---

### 💰 When is accuracy “good enough”?

Always tie eval scores to **business cost**:

- Cost of failure
- Cost of escalation
- Cost of human review
- Customer churn risk

Then design technical guardrails:
- confidence checks
- human fallback
- clarification loops

LLMs don’t need 99% — they need **economically safe accuracy**.

---

### ✅ Big Takeaways

- Prompting is a measurable engineering discipline  
- Always optimize with evals  
- Fix context problems with RAG  
- Fix behavior problems with fine-tuning  
- Stack techniques strategically  
- Tie quality to business impact  

---

### 🔗 Core Resources

https://platform.openai.com/docs/guides/optimizing-llm-accuracy  
https://platform.openai.com/docs/guides/prompt-engineering  
https://platform.openai.com/docs/guides/reasoning  
https://platform.openai.com/docs/guides/reasoning-best-practices  
https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide

------------------------------------------------------------------------

## 🤖 3. Agents, Tools & Context Engineering

This session focused on **building reliable autonomous systems** using tools, MCP, and disciplined context management.

---

### 🔌 Model Context Protocol (MCP)

MCP is an **open standard for connecting AI systems to external tools and data sources**.

Think of it as:

**USB‑C for AI systems** → one protocol to plug agents into databases, files, APIs, workflows, and apps.

**What MCP enables**
- Agents call enterprise tools & databases directly  
- Standardized tool interfaces across vendors  
- Faster agent development without custom glue code  
- Interoperable ecosystems of tools  

**Why it matters**
- Removes brittle custom integrations  
- Scales agent capabilities cleanly  
- Makes AI systems composable  

**Mental model**

Agent ↔ MCP Server ↔ Tools / Data / Workflows

---

### ⚡ Realtime & Agent Orchestration

- Multi‑step planning  
- Tool calling loops  
- Long‑running workflows  
- Streaming reasoning  
- Modular agent design  

Used for automation, analytics, dev agents, and multimodal systems.

### 📦 MCP_IN_A_BOX — MCP Hands-on Sandbox  
A ready-to-run playground for building MCP servers and connecting agents to real tools and data sources.  
Great for understanding end-to-end MCP flows without custom infrastructure. \
https://github.com/shikhar-cyber/MCP_IN_A_BOX  

### ⚡ openai-realtime-agents — Real-Time Agent Orchestration  
Reference implementations for multi-step, tool-using, streaming AI agents built on OpenAI APIs.  
Shows planning → tool calls → feedback loops for production-style autonomous systems. \
https://github.com/openai/openai-realtime-agents  

### 📖 openai-knowledge-retrieval — Knowledge + Retrieval Example  
A reference implementation for building **retrieval-augmented systems** with integrated search, citations, and evaluation hooks.  
Useful for learning how to wire retrieval sources (vector stores, search) into LLM apps with scoring and QA. \
https://github.com/openai/openai-knowledge-retrieval

---

### 🧠 Context Engineering (Agent Memory)

Goal: preserve what matters without blowing token limits.

Main risks:
- Context bloat  
- Hallucination carry‑forward  
- Tool confusion  
- Rising cost  

### ✂️ Context Trimming

Keep last **N user turns** only.

Best for:
- Tool workflows  
- Independent tasks  
- Predictable state  

### 🗜️ Context Summarization

Compress older history into structured summaries.

Best for:
- Long reasoning  
- Planning  
- RAG‑heavy flows  

### 1. Never treat summaries as facts
Label memory items:

- TOOL-VERIFIED  
- USER-CLAIM  
- UNVERIFIED  
- SUPERSEDED  

Uncertainty stays visible.

### 2. Resolve conflicts & recency
- Newer data overrides old  
- Conflicts are explicit  
- Wrong info is not reused silently  

### 3. Keep recent turns raw
Only compress older context.

### 4. Gate critical actions
IDs, amounts, configs, dates require:
- tool verification OR  
- explicit user confirmation

### 5. Prefer raw tool outputs
Extract facts only when clearly verified.

### Minimal Safety Rules

1. Every summary item has certainty label  
2. Conflicts are marked  
3. Old info can be superseded  
4. Critical fields must be verified  
5. Never summarize summaries blindly  

**One-line takeaway:**  
Treat memory as uncertain state — verify before acting.

Long context ≠ good context.

### 🧪 Evals for Memory

- Replay conversations  
- Judge summary accuracy  
- Track entity/tool correctness  
- Detect regressions  

### 🔗 Resources

https://modelcontextprotocol.io  
https://cookbook.openai.com/examples/agents_sdk/session_memory



------------------------------------------------------------------------

## 🚦 4. Production Optimization

This session focuses on taking an LLM prototype to production: access control, scaling, latency/cost control, safety, and rate-limit resilience.

### 4.1 Scaling the solution architecture

- **Horizontal scaling**: add more stateless workers; use a queue for bursty traffic.
- **Vertical scaling**: scale up a single node when concurrency is low but per-request work is heavy.
- **Caching**:
  - Cache **model outputs** for repeatable prompts (e.g. ticket categorization).
  - Cache **retrieval** (embeddings, top-k hits) with invalidation on content changes.
- **Load balancing**: distribute requests across workers; keep retries idempotent.

### 4.2 Latency optimization

1. **Process tokens faster**
   - Use a **smaller model** where quality allows; reserve large models for high-stakes generation.
2. **Generate fewer tokens**
   - Tight `max_output_tokens`; add **stop sequences**; ask for concise output.
   - For structured outputs: shorten field names / schemas when safe.
3. **Use fewer input tokens**
   - Prune RAG context; clean HTML; de-duplicate tool outputs.
4. **Make fewer requests**
   - Combine steps into one call when sequential dependencies are small (e.g., “extract intent + generate answer”).
5. **Parallelize**
   - Run independent steps concurrently (e.g., retrieval + lightweight classification).
6. **Make users wait less**
   - Use **streaming** for faster time-to-first-token.
   - If post-processing is required, process streamed chunks on the backend.
7. **Don’t default to an LLM**
   - Hard-code constrained responses; use deterministic rules for simple routing; precompute for fixed inputs.

### 4.3 Batch API (async bulk jobs)

Use Batch when you do **not** need immediate responses (e.g., evals, dataset classification, embedding corpora).

- **How it works**
  1. Build a `.jsonl` file (one request per line) targeting a single endpoint/model.
  2. Upload via **Files API** with `purpose="batch"`.
  3. Create a batch referencing the uploaded file (`completion_window="24h"`).
  4. Poll status; download results via the output file ID.
- **Why Batch**
  - **~50% cost discount** vs synchronous calls.
  - Separate, **higher** rate-limit pool.
- **Operational notes**
  - Output order may differ; use `custom_id` to join results.
  - Input file limits: up to **50k requests** and **200MB** per batch (endpoint-specific constraints apply).

### 4.4 Rate limits and reliability patterns

- **Rate-limit dimensions**
  - **RPM / RPD**: requests per minute/day
  - **TPM / TPD**: tokens per minute/day
  - **IPM**: images per minute (if using image inputs)
- **Key behaviors**
  - Limits are primarily **org/project** scoped (not per end-user).
  - Some model families share pooled limits.
  - Batch jobs use a separate quota (good for bulk)
- **Mitigations**
  - **Exponential backoff + jitter** On 429 errors.
  - Keep `max_output_tokens` close to expected output size.
  - If RPM-bound but TPM available: **pack multiple tasks** into one request (careful with quality).
  - For bulk offline work: prefer **Batch API**.

### 4.5 Safety best practices (production defaults)

- **Moderation**: run user input and/or model output through Moderation for high-risk domains.
- **Adversarial testing**: prompt-injection attempts, jailbreaks, long-context attacks.
- **Human-in-the-loop**: review gates for high-stakes actions (finance, legal, medical, destructive ops).
- **Constrain inputs/outputs**
  - Validate inputs (schemas, allow-lists), cap length, cap tool-call surfaces.
  - Cap output tokens; prefer deterministic backends for known answers.
- **Safety identifiers**
  - Send a per-user (hashed) identifier to help detect abuse patterns without sending PII.

🔗 References
- https://platform.openai.com/docs/guides/production-best-practices
- https://platform.openai.com/docs/guides/latency-optimization
- https://platform.openai.com/docs/guides/batch
- https://platform.openai.com/docs/guides/rate-limits
- https://platform.openai.com/docs/guides/safety-best-practices

------------------------------------------------------------------------

## 💻 5. Codex & Developer Acceleration

**What “Codex” is (in OpenAI Platform context)**
- Code-specialized models + workflows optimized for editing, refactoring, reviewing, and generating patches with high precision.
- Best used when you want *file-aware* changes (diffs/patches), deterministic steps, and repeatable automation around a repo.

**1) CLI workflows**
- Use for: local repo edits, quick refactors, batch fixes across many files.
- Pattern: `plan → patch → validate` (run tests/linters after each patch).
- Guardrails: run in a sandboxed workspace; never pass secrets; keep outputs as diffs/PR-ready commits.

**2) SDK integrations**
- Use for: embedding Codex into internal developer tools (e.g., “PR reviewer bot”, “migration assistant”, “lint fixer”).
- Typical loop: fetch context (files/PR diff) → ask for structured change plan → generate patch/diff → run checks → post results.

**3) Automated refactoring**
- Best for: mechanical changes (rename, API migration, formatting), repetitive bug patterns, “apply the same fix everywhere”.
- Reliability tricks:
  - Ask for *targeted edits* (file list + exact symbols) instead of broad rewrites.
  - Prefer patches/diffs over “paste full file”.
  - Verify with unit tests + type checks; fail closed if checks fail.

**4) Execution plans (PLANS.md / step list)**
- Capture: goals, non-goals, constraints, files touched, risk notes, validation steps.
- Treat plan as a contract: model must not deviate without explicitly updating the plan.
- Minimal template:
  - Goal → Assumptions → Steps → Files → Validation → Rollback

🔗 References
- https://cookbook.openai.com/topic/codex
------------------------------------------------------------------------
