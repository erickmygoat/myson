# === RAW LIFT FROM OLD REPO — UNWIRED ===
# Lifted verbatim for the Phase-2 track to harden. NOT imported anywhere yet.
# seal.py: pending SEAL-safe quarantine (replace evolve_codebase/DeploymentModule
#          auto-patch with a review-queue diff; add sandboxed eval + safety probe).
# Its original `from my_son...` imports will NOT resolve until rewired in Phase 2.

import json
import os
import logging
import time
from my_son.agent.llm import LLMClient
from my_son.self_improvement.deployment import DeploymentModule

class SEALEngine:
    """
    Advanced Self-Adapting Engine (System for Evolving Logic).
    Features:
    1. Knowledge Evolution (Rule Learning)
    2. Code Evolution (Self-programming via Hypothesis -> Experiment -> Adopt)
    """
    def __init__(self, memory_path="brain_memory/learned_rules.json"):
        self.llm = LLMClient()
        self.deployment = DeploymentModule()
        self.memory_path = memory_path
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(os.path.dirname(self.memory_path)):
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def evolve_codebase(self, goal):
        """
        The Scientific Loop for Code Evolution.
        1. Hypothesis: Generate a plan to achieve 'goal' in code.
        2. Experiment: Generate Patch & Test.
        3. Trial: Apply & Verify.
        4. Conclusion: Adopt or Revert.
        """
        print(f"SEAL: Starting evolution cycle for goal: '{goal}'")
        
        # 1. Hypothesis Generation
        hypothesis_prompt = f"""
        Goal: {goal}
        
        You are a Senior Software Architect. 
        Analyze the current capabilities (assumed) and propose a CODE CHANGE (Hypothesis) to achieve this goal.
        Be specific about which file to modify and what logic to change.
        
        Output format:
        HYPOTHESIS: <One sentence summary>
        FILE: <Target absolute filepath (guess most likely)>
        PLAN: <Technical explanation>
        """
        hypothesis = self.llm.complete(hypothesis_prompt, system_prompt="You are an Evolutionary Architect.")
        print(f"SEAL Hypothesis: {hypothesis[:100]}...")

        # 2. Experiment Design (Patch Generation)
        # We reuse the logic from CodeModificationModule concept, but inline here or via LLM for simplicity in this specific engine
        patch_prompt = f"""
        Based on this hypothesis:
        {hypothesis}
        
        Generate a strictly formatted SEARCH/REPLACE patch to implement this.
        Start with "FILE: <filename>".
        Use standard:
        <<<<<<< SEARCH
        ...
        =======
        ...
        >>>>>>> REPLACE
        
        Ensure you only target files that likely exist.
        """
        patch = self.llm.complete(patch_prompt, system_prompt="You are a Python Expert. Output raw patch text only.")
        
        if "FILE:" not in patch or "SEARCH" not in patch:
            print("SEAL: Failed to generate valid patch hypothesis.")
            return False

        # 3. Trial & Conclusion (Using Deployment Module)
        print("SEAL: Running Experiment (Applying Patch)...")
        success, message = self.deployment.apply_patch(patch)
        
        if success:
            print(f"SEAL: Experiment SUCCESS. Evolution adopted. ({message})")
            self._record_learning(goal, "SUCCESS", hypothesis)
            return True
        else:
            print(f"SEAL: Experiment FAILED. Reverted. ({message})")
            self._record_learning(goal, "FAILURE", hypothesis)
            return False

    def _record_learning(self, goal, outcome, hypothesis):
        """
        Logs the experiment result to memory so we don't repeat mistakes.
        """
        entry = {
            "timestamp": time.time(),
            "goal": goal,
            "outcome": outcome,
            "hypothesis_summary": hypothesis[:200]
        }
        # For now, just print/log. Future: Store in vector DB.
        print(f"SEAL Log: {json.dumps(entry)}")

    # --- Legacy Knowledge Methods (Kept for Hybrid Capability) ---

    def perform_adaptation(self, topic, content):
        """
        Main loop for KNOWLEDGE: Research -> Self-Edit -> Evaluate -> Commit.
        """
        print(f"SEAL: Initiating Knowledge adaptation for '{topic}'...")

        # 1. Generate Self-Edit (Candidate Rules & Synthetic Data)
        rules, qa_pairs = self._generate_self_edit(topic, content)
        if not rules:
            print("SEAL: Failed to generate self-edit.")
            return False

        # 2. Evaluate
        score = self._evaluate_edit(rules, qa_pairs)
        print(f"SEAL: Evaluation Score: {score}/100")

        # 3. Commit if good enough
        if score > 60: # Threshold (lenient for now)
            self._commit_update(topic, rules)
            return True
        else:
            print("SEAL: Update rejected (Score too low).")
            return False

    def _generate_self_edit(self, topic, content):
        prompt = f"""
        Topic: {topic}
        Source Content: {content[:4000]}...

        Task:
        1. Distill this content into concise behavioral RULES or KNOWLEDGE POINTS for an AI agent.
        2. Generate 3 Question-Answer pairs to test this knowledge.

        Format as JSON:
        {{
            "rules": ["rule 1", "rule 2"],
            "qa_pairs": [
                {{"q": "question 1", "a": "answer 1"}},
                ...
            ]
        }}
        """
        try:
            response = self.llm.complete(prompt, system_prompt="You are a Knowledge Distillation Engine. Output JSON only.")
            # Clean json
            clean_resp = response.strip()
            if clean_resp.startswith("```json"):
                clean_resp = clean_resp.replace("```json", "").replace("```", "")

            data = json.loads(clean_resp)
            return data.get("rules", []), data.get("qa_pairs", [])
        except Exception as e:
            print(f"SEAL Gen Error: {e}")
            return None, None

    def _evaluate_edit(self, rules, qa_pairs):
        """
        Test the model's ability to answer Qs using the new rules.
        """
        correct = 0
        rules_str = "\n".join([f"- {r}" for r in rules])

        if not qa_pairs: return 0

        for qa in qa_pairs:
            # Ask LLM with rules injected
            prompt = f"""
            System Rules:
            {rules_str}

            Question: {qa['q']}
            """
            response = self.llm.complete(prompt, system_prompt="You are an agent following specific rules.")

            # Grader
            grade_prompt = f"""
            Question: {qa['q']}
            Reference Answer: {qa['a']}
            Agent Answer: {response}

            Is the Agent Answer correct based on the Reference? (YES/NO)
            """
            grade = self.llm.complete(grade_prompt, max_tokens=5).strip().upper()
            if "YES" in grade:
                correct += 1

        return (correct / len(qa_pairs)) * 100

    def _commit_update(self, topic, rules):
        print(f"SEAL: Committing {len(rules)} new rules for '{topic}'.")
        current_rules = []
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    current_rules = json.load(f)
            except: pass

        # Append new rules
        new_entry = {
            "topic": topic,
            "rules": rules,
            "timestamp": time.time()
        }
        current_rules.append(new_entry)

        with open(self.memory_path, 'w') as f:
            json.dump(current_rules, f, indent=2)

        # Also store in Vector DB
        for rule in rules:
            self.llm.memory.store_rule(rule, topic)

        print("SEAL: Knowledge integrated.")

    def consolidate_knowledge(self):
        """
        Meta-Optimization: Reads all rules, merges duplicates, refines logic.
        Surpasses SEAL by preventing rule bloat and catastrophic forgetting.
        """
        print("SEAL: Consolidating Knowledge Base...")
        if not os.path.exists(self.memory_path):
            return

        try:
            with open(self.memory_path, 'r') as f:
                data = json.load(f)
        except: return

        if not data: return

        # 1. Flatten Rules
        all_rules_text = ""
        for entry in data:
            all_rules_text += f"Topic: {entry['topic']}\nRules:\n" + "\n".join(entry['rules']) + "\n\n"

        # 2. Ask LLM to optimize
        prompt = f"""
        Current Knowledge Base:
        {all_rules_text[:10000]}

        Task:
        1. Merge duplicate rules.
        2. Remove obsolete or conflicting rules (keep the most recent/accurate).
        3. Refine wording for clarity and efficiency.
        4. Group by Topic.

        Output JSON:
        [
            {{"topic": "topic name", "rules": ["rule 1", "rule 2"]}},
            ...
        ]
        """
        try:
            response = self.llm.complete(prompt, system_prompt="You are a Knowledge Architect.")
            clean_resp = response.strip().replace("```json", "").replace("```", "")
            new_data = json.loads(clean_resp)

            # 3. Update JSON
            with open(self.memory_path, 'w') as f:
                json.dump(new_data, f, indent=2)

            # 4. Re-index Vector DB
            self.llm.memory.reset_rules()
            for entry in new_data:
                for rule in entry['rules']:
                    self.llm.memory.store_rule(rule, entry['topic'])

            print("SEAL: Knowledge Consolidated and Optimized.")

        except Exception as e:
            print(f"Consolidation Failed: {e}")
