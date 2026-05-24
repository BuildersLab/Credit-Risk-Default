# Decision log: Credit Risk Default

Every significant technical decision is recorded here with context and rationale.
Add a row every time the team makes a non-obvious choice. Resolved disagreements belong here too.

---

## Dataset selection

**Decision:** Credit risk data - TBD
**Alternatives considered:** TODO
**Rationale:** TODO

---

## Primary evaluation metric

**Decision:** PR-AUC
**Alternatives considered:** TODO
**Rationale:** TODO — explain why this metric fits the class distribution and business goal for Credit Risk Default

---

## Train/validation split strategy

**Decision:** TODO (time-based / stratified random / other)
**Alternatives considered:** TODO
**Rationale:** TODO — explain leakage risk and why this split reflects real-world model deployment

---

## LLM provider (if using Gemini)

**Decision:** Google Gemini API
**Alternatives considered:** OpenAI GPT-4o, Anthropic Claude, local Llama
**Rationale:** Free tier sufficient for demo use. Google AI Studio simplifies key setup for the team.

---

## Gemini role

**Decision:** Advisory only. Gemini explains predictions. It does not make or modify them.
**Rationale:** Automated decisions affecting people must be traceable to a deterministic auditable system. Gemini is a communication layer, not a decision layer.

---

## Demo platform

**Decision:** Replit
**Alternatives considered:** Streamlit Cloud, Hugging Face Spaces
**Rationale:** Public URL, no login required, easy to share with employers.

---

## Prompts in a separate file

**Decision:** All Gemini prompt templates in `src/prompts.py`
**Rationale:** Bintou Ba, Marienne Dosso, Lynda Allepo can iterate on prompts independently from the API code owned by Divyanshi kashyap.

---

<!-- Add new decisions below as they are made -->
