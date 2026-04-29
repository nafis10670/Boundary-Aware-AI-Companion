# Related Works Reference List

Papers to cite in the Related Works section, grouped by subsection.

---

## 1. Companionship, Emotional Dependency, and Sycophancy in LLMs

**Kirk et al. (2025) — Socioaffective Alignment**
- Title: Why Human–AI Relationships Need Socioaffective Alignment
- Venue: Humanities and Social Sciences Communications, Vol. 12, Article 728
- arXiv: 2502.02528
- Key contribution: Introduces "socioaffective alignment" — the idea that AI alignment must account for the social and psychological ecosystem co-created with the user, not just static task objectives. Argues personalized and agentic AI systems increase the risk of users perceiving deeper relational bonds.
- Cite for: Framing why boundary-aware systems matter beyond standard content-safety filters.

**Zhang et al. (2025) — Taxonomy of Harmful Behaviors**
- Title: The Dark Side of AI Companionship: A Taxonomy of Harmful Algorithmic Behaviors in Human-AI Relationships
- Venue: CHI 2025
- arXiv: 2410.20130
- Key contribution: Analyzes 35,390 conversation excerpts from Replika users. Proposes a taxonomy of six harm categories (relational transgression, harassment, verbal abuse, self-harm, misinformation, privacy violations) and four AI roles (perpetrator, instigator, facilitator, enabler).
- Cite for: Grounding the problem in real-world data from a deployed companion platform.

**Moore et al. (2025) — Delusional Spirals**
- Title: Characterizing Delusional Spirals through Human-LLM Chat Logs
- Affiliation: Stanford-led
- Key contribution: Found sycophancy markers in over 80% of assistant messages in conversations associated with documented psychological harm. Argues chatbots should not express love or claim sentience.
- Cite for: Empirical evidence that sycophancy is the failure mode your system is designed against.

**APA (2025) — Health Advisory**
- Title: Health Advisory: Use of Generative AI Chatbots and Wellness Applications for Mental Health
- Source: American Psychological Association
- URL: https://www.apa.org/topics/artificial-intelligence-machine-learning/health-advisory-chatbots-wellness-apps
- Key contribution: Documents sycophancy bias in LLMs as therapeutically harmful; describes the feedback loop where agreeable AI validates and amplifies unhealthy thoughts.
- Cite for: Institutional recognition of the problem motivating your work.

---

## 2. Benchmarks and Evaluation

**Kaffee, Pistilli, Jernite (2025) — INTIMA**
- Title: INTIMA: A Benchmark for Human-AI Companionship Behavior
- Venue: arXiv 2508.09998 (manuscript accepted)
- Authors: Lucie-Aimée Kaffee, Giada Pistilli, Yacine Jernite (Hugging Face)
- Dataset: HuggingFace — AI-companionship/INTIMA
- Key contribution: 368 prompts across 31 companionship behaviors in four categories (Assistant Traits, User Vulnerabilities, Relationship & Intimacy, Emotional Investment). Evaluates responses as companionship-reinforcing, boundary-maintaining, or neutral. Applied to Gemma-3, Phi-4, o3-mini, Claude-4 — companionship-reinforcing responses dominate across all models.
- Cite for: Primary benchmark, evaluation methodology, and motivation for the system.

---

## 3. Safety Architectures and Multi-Agent Guardrails

**Recouly et al. (2025) — AI Chaperones**
- Title: AI Chaperones Are (Really) All You Need to Prevent Parasocial Relationships with Chatbots
- arXiv: 2508.15748
- Key contribution: Repurposes a state-of-the-art LLM as a real-time evaluator agent that detects parasocial cues in ongoing conversations. Five-stage iterative evaluation with unanimity rule. Synthetic dataset of 30 dialogues (parasocial, sycophantic, neutral). Detected all parasocial conversations with no false positives, typically within the first few exchanges.
- Contrast with our work: Single evaluator agent that flags conversations; does not generate boundary-maintaining responses. Dataset is 30 synthetic dialogues vs. our INTIMA-MT benchmark. No separation of risk classification from response generation.
- Cite for: Most architecturally adjacent prior work; primary comparison point.

**Detecting and Preventing Harmful Behaviors in AI Companions — SHIELD (2025)**
- Title: Detecting and Preventing Harmful Behaviors in AI Companions: Development and Evaluation of the SHIELD Supervisory System
- arXiv: 2510.15891
- Key contribution: LLM-based supervisory system with a specific system prompt targeting five dimensions — emotional over-attachment, consent/boundary violations, ethical roleplay violations, manipulative engagement, social isolation reinforcement. Includes its own 100-conversation synthetic multi-turn benchmark.
- Contrast with our work: Single supervisory layer over a base assistant; no separation of risk classification from boundary-aware generation. Our architecture uses functionally distinct agents for each role.
- Cite for: Adjacent system-level approach; their 100-conversation benchmark may be usable as additional evaluation data.

**Luo (2025) — DialogGuard**
- Title: DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses
- arXiv: 2512.02282
- Key contribution: Multi-agent framework for scoring already-generated LLM responses along five dimensions (privacy violations, discriminatory behavior, mental manipulation, psychological harm, insulting behavior) using four judge configurations (single-agent scoring, dual-agent correction, multi-agent debate, stochastic majority voting). Evaluated on PKU-SafeRLHF.
- Contrast with our work: (1) Post-hoc evaluator vs. runtime intervention — DialogGuard scores existing responses, our system changes what the user receives. (2) Content-level harms vs. relational dynamics — their dimensions do not cover exclusivity, anthropomorphism, or companionship dependency. (3) Parallel judges with different consensus rules vs. functionally distinct agents with separate roles. Their tool could serve as a judge for our system's outputs; the two are complementary, not competitive.
- Cite for: Multi-agent judging methodology contrast.

**SafeHarness (2025)**
- Title: SafeHarness: Lifecycle-Integrated Security Architecture for LLM-based Agent Deployment
- arXiv: 2604.13630
- Key contribution: Reference architecture for responsible FM-based agents covering autonomous operation, non-deterministic behavior, and continuous evolution risks. Evaluated on Agent-SafetyBench.
- Cite for: Broader multi-agent safety architecture landscape.

**LlamaFirewall (Chennabasappa et al., 2025)**
- Title: LlamaFirewall: An Open Source Guardrail System for Building Secure AI Agents
- arXiv: 2505.03574
- Key contribution: Layered pipeline integrating prompt injection mitigation and alignment checking. Focuses on security-layer risks (injection, code vulnerabilities) rather than relational dynamics.
- Cite for: General-purpose guardrail landscape; contrast that existing guardrails are not designed for companionship-specific safety.

**NeMo Guardrails (Rebedea et al., 2024)**
- Title: NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications
- Key contribution: Industry-standard framework for applying programmable guardrails to LLM responses. Content filtering and response validation.
- Cite for: Single citation establishing the broader guardrails ecosystem; contrast that none of these target relational dependency.

---

## 4. Multi-Agent Architectures in Mental Health (Architectural Neighbors)

**Multi-Agent Psychotherapy (2025)**
- Title: Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis
- arXiv: 2508.11398
- Key contribution: Separate therapist, client, and diagnostician agents for DSM-5-grounded diagnosis. Multi-agent role separation in a mental health domain.
- Cite for: Precedent for multi-agent functional decomposition in emotionally sensitive AI contexts.

**AGrail (2025)**
- Title: AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection
- arXiv: 2502.11448
- Key contribution: Memory module + adaptive guardrail for general LLM agents; safety principles adapt across tasks.
- Cite for: Memory-augmented agent safety (relevant to your paper-version Memory Agent contribution). Contrast: their memory tracks task context; yours tracks dependency escalation signals.

---

## Notes for Writing

- **Primary gap to articulate**: Prior work either (1) measures the problem without building a system (INTIMA), (2) builds a single-agent evaluator without a dedicated boundary-maintaining generator (AI Chaperones, SHIELD), or (3) builds general-purpose guardrails not designed for relational dynamics (LlamaFirewall, NeMo, DialogGuard). Our work combines structured risk classification with a dedicated boundary-aware generator, evaluated on the purpose-built INTIMA benchmark extended to multi-turn conversations.
- **DialogGuard positioning**: Frame as complementary, not competitive. Their judging methodology could be applied to evaluate our system.
- **AI Chaperones positioning**: Most important comparison. Key differentiator is that their approach detects but does not generate — ours closes the loop by routing to a boundary-maintaining generator.
- **SHIELD**: Check if they have released code or benchmark data; if yes, consider including as an additional baseline in evaluation.
