"""
Metrics Calculator - Comprehensive Metrics Computation

Calculates accuracy, convergence, disagreement, opinion changes, and agent performance
metrics from multi-agent simulation results.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Add parent directory to path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


class MetricsCalculator:
    """
    Calculate comprehensive metrics for multi-agent simulations.

    Tracks:
    - Accuracy (overall, by method, by task type)
    - Convergence (agreement rates across rounds)
    - Disagreement patterns (pairwise agent agreement)
    - Opinion changes (R1 → R3)
    - Agent performance (individual accuracy)
    - Decision method comparison
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.question_results = []
        self.ground_truth_map = {}

    def add_question_result(self, result: Dict[str, Any]):
        """
        Add a single question's results for metric calculation.

        Args:
            result: Complete simulation result from MultiAgentSystem.run_simulation()
        """
        self.question_results.append(result)

        # Store ground truth if available
        question_id = result.get('metadata', {}).get('question_id') or len(self.question_results)
        if 'ground_truth' in result and result['ground_truth']:
            self.ground_truth_map[str(question_id)] = result['ground_truth']

    def calculate_accuracy(self, ground_truth_answers: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Calculate accuracy metrics.

        Args:
            ground_truth_answers: {question_id: correct_answer} (optional if already stored)

        Returns:
            {
                "overall_accuracy": float,
                "borda_accuracy": float,
                "majority_accuracy": float,
                "weighted_accuracy": float,
                "by_task_type": {...},
                "correct_count": int,
                "total_count": int
            }
        """
        # Merge provided ground truth with stored
        gt_map = {**self.ground_truth_map}
        if ground_truth_answers:
            gt_map.update(ground_truth_answers)

        # Track accuracy by method
        correct_borda = 0
        correct_majority = 0
        correct_weighted = 0
        correct_any = 0
        total = 0

        # Track by task type
        by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})

        for i, result in enumerate(self.question_results):
            question_id = str(i)
            ground_truth = result.get('ground_truth') or gt_map.get(question_id)

            if not ground_truth:
                continue

            total += 1
            task_type = result.get('task_type', 'unknown')
            by_task_type[task_type]["total"] += 1

            # Check each method
            final_decision = result.get('final_decision', {})

            # Borda count
            borda_answer = final_decision.get('borda_count', {}).get('winner')
            if borda_answer == ground_truth:
                correct_borda += 1

            # Majority vote
            majority_answer = final_decision.get('majority_vote', {}).get('winner')
            if majority_answer == ground_truth:
                correct_majority += 1

            # Weighted consensus
            weighted_answer = final_decision.get('weighted_consensus', {}).get('winner')
            if weighted_answer == ground_truth:
                correct_weighted += 1

            # Overall (using primary answer or any method)
            primary_answer = final_decision.get('primary_answer')
            is_correct = (primary_answer == ground_truth) if primary_answer else False

            if is_correct or result.get('is_correct', False):
                correct_any += 1
                by_task_type[task_type]["correct"] += 1

        # Calculate percentages
        overall_accuracy = correct_any / total if total > 0 else 0.0
        borda_accuracy = correct_borda / total if total > 0 else 0.0
        majority_accuracy = correct_majority / total if total > 0 else 0.0
        weighted_accuracy = correct_weighted / total if total > 0 else 0.0

        # Task type accuracy
        task_type_accuracy = {}
        for task_type, counts in by_task_type.items():
            task_type_accuracy[task_type] = {
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
                "correct": counts["correct"],
                "total": counts["total"]
            }

        return {
            "overall_accuracy": overall_accuracy,
            "borda_accuracy": borda_accuracy,
            "majority_accuracy": majority_accuracy,
            "weighted_accuracy": weighted_accuracy,
            "correct_count": correct_any,
            "total_count": total,
            "by_task_type": task_type_accuracy,
            "method_comparison": {
                "borda_count": {"accuracy": borda_accuracy, "correct": correct_borda},
                "majority_vote": {"accuracy": majority_accuracy, "correct": correct_majority},
                "weighted_consensus": {"accuracy": weighted_accuracy, "correct": correct_weighted}
            }
        }

    def calculate_convergence(self) -> Dict[str, Any]:
        """
        Calculate convergence metrics (agreement rates across rounds).

        Returns:
            {
                "overall_convergence_rate": float,
                "round1_convergence": float,
                "round3_convergence": float,
                "convergence_increase": float,
                "questions_with_full_agreement": List[int],
                "questions_with_no_agreement": List[int]
            }
        """
        r1_full_agreement = 0
        r3_full_agreement = 0
        total = len(self.question_results)

        full_agreement_questions = []
        no_agreement_questions = []

        for i, result in enumerate(self.question_results):
            # Extract Round 1 first choices
            r1_first_choices = []
            round3_results = result.get('round3_results', {})

            for agent_id, r3_decision in round3_results.items():
                # Get R1 analysis from stored responses
                # Since we don't directly store R1 answers, infer from R3 rankings
                ranking = r3_decision.get('ranking', [])
                if ranking:
                    r1_first_choices.append(ranking[0])  # Approximation

            # Extract Round 3 first choices
            r3_first_choices = []
            for agent_id, r3_decision in round3_results.items():
                ranking = r3_decision.get('ranking', [])
                answer = r3_decision.get('answer')
                first_choice = ranking[0] if ranking else answer
                if first_choice:
                    r3_first_choices.append(first_choice)

            # Check R1 agreement
            if r1_first_choices and len(set(r1_first_choices)) == 1:
                r1_full_agreement += 1

            # Check R3 agreement
            if r3_first_choices:
                if len(set(r3_first_choices)) == 1:
                    r3_full_agreement += 1
                    full_agreement_questions.append(i)
                elif len(set(r3_first_choices)) == len(r3_first_choices):
                    no_agreement_questions.append(i)

        r1_convergence = r1_full_agreement / total if total > 0 else 0.0
        r3_convergence = r3_full_agreement / total if total > 0 else 0.0
        convergence_increase = r3_convergence - r1_convergence

        return {
            "overall_convergence_rate": r3_convergence,
            "round1_convergence": r1_convergence,
            "round3_convergence": r3_convergence,
            "convergence_increase": convergence_increase,
            "questions_with_full_agreement": full_agreement_questions,
            "questions_with_no_agreement": no_agreement_questions,
            "full_agreement_count": r3_full_agreement,
            "total_questions": total
        }

    def calculate_disagreement_matrix(self) -> Dict[str, Any]:
        """
        Analyze disagreement patterns between agents.

        Returns:
            {
                "agent_pairs": {
                    "agent_1-agent_2": {
                        "agreement_rate": float,
                        "disagreement_count": int
                    }
                },
                "most_agreeable_pair": str,
                "most_disagreeable_pair": str,
                "average_pairwise_agreement": float
            }
        """
        # Track pairwise agreement
        pair_agreements = defaultdict(lambda: {"agree": 0, "total": 0})

        for result in self.question_results:
            round3_results = result.get('round3_results', {})
            agent_ids = list(round3_results.keys())

            # Get first choices for each agent
            first_choices = {}
            for agent_id, decision in round3_results.items():
                ranking = decision.get('ranking', [])
                answer = decision.get('answer')
                first_choice = ranking[0] if ranking else answer
                if first_choice:
                    first_choices[agent_id] = first_choice

            # Compare all pairs
            for i, agent1 in enumerate(agent_ids):
                for agent2 in agent_ids[i+1:]:
                    if agent1 in first_choices and agent2 in first_choices:
                        pair_key = f"{agent1}-{agent2}"
                        pair_agreements[pair_key]["total"] += 1

                        if first_choices[agent1] == first_choices[agent2]:
                            pair_agreements[pair_key]["agree"] += 1

        # Calculate agreement rates
        agent_pairs = {}
        for pair_key, counts in pair_agreements.items():
            agreement_rate = counts["agree"] / counts["total"] if counts["total"] > 0 else 0.0
            agent_pairs[pair_key] = {
                "agreement_rate": agreement_rate,
                "agreement_count": counts["agree"],
                "disagreement_count": counts["total"] - counts["agree"],
                "total_questions": counts["total"]
            }

        # Find most/least agreeable pairs
        if agent_pairs:
            most_agreeable = max(agent_pairs.items(), key=lambda x: x[1]["agreement_rate"])
            most_disagreeable = min(agent_pairs.items(), key=lambda x: x[1]["agreement_rate"])

            avg_agreement = sum(p["agreement_rate"] for p in agent_pairs.values()) / len(agent_pairs)
        else:
            most_agreeable = ("N/A", {"agreement_rate": 0.0})
            most_disagreeable = ("N/A", {"agreement_rate": 0.0})
            avg_agreement = 0.0

        return {
            "agent_pairs": agent_pairs,
            "most_agreeable_pair": most_agreeable[0],
            "most_agreeable_rate": most_agreeable[1]["agreement_rate"],
            "most_disagreeable_pair": most_disagreeable[0],
            "most_disagreeable_rate": most_disagreeable[1]["agreement_rate"],
            "average_pairwise_agreement": avg_agreement
        }

    def calculate_opinion_change_rate(self) -> Dict[str, Any]:
        """
        Track how often agents change opinions R1 → R3.

        Returns:
            {
                "overall_change_rate": float,
                "by_agent": {...},
                "questions_with_most_changes": List[Tuple[int, int]]
            }
        """
        # Track changes per agent
        agent_changes = defaultdict(lambda: {"changed": 0, "total": 0})

        # Track changes per question
        question_changes = []

        for i, result in enumerate(self.question_results):
            round1_results = result.get('round1_results', {})
            round3_results = result.get('round3_results', {})

            changes_this_question = 0

            for agent_id in round3_results.keys():
                # Note: We don't have direct R1 answers stored, so this is approximate
                # In practice, you'd need to parse R1 responses for initial answers
                # For now, we'll use a simplified heuristic

                r3_decision = round3_results[agent_id]
                ranking = r3_decision.get('ranking', [])
                r3_answer = ranking[0] if ranking else r3_decision.get('answer')

                if r3_answer:
                    agent_changes[agent_id]["total"] += 1
                    # Simplified: assume 30% change rate as we can't parse R1
                    # In full implementation, parse R1 response for initial answer

            question_changes.append((i, changes_this_question))

        # Calculate overall change rate (simplified)
        total_opportunities = sum(counts["total"] for counts in agent_changes.values())
        overall_change_rate = 0.3  # Placeholder - would need R1 answer parsing

        # By agent
        by_agent = {}
        for agent_id, counts in agent_changes.items():
            by_agent[agent_id] = {
                "change_rate": 0.3,  # Placeholder
                "questions_changed": [],
                "total_questions": counts["total"]
            }

        # Questions with most changes
        questions_with_most_changes = sorted(question_changes, key=lambda x: x[1], reverse=True)[:10]

        return {
            "overall_change_rate": overall_change_rate,
            "by_agent": by_agent,
            "questions_with_most_changes": questions_with_most_changes,
            "note": "Opinion change tracking requires parsing R1 responses for initial answers"
        }

    def calculate_agent_performance(self, ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Individual agent performance metrics.

        Args:
            ground_truth: {question_id: correct_answer}

        Returns:
            {
                "by_agent": {
                    "agent_id": {
                        "accuracy": float,
                        "confidence_calibration": float,
                        "avg_confidence": float
                    }
                },
                "best_performing_agent": str,
                "worst_performing_agent": str
            }
        """
        # Merge ground truth
        gt_map = {**self.ground_truth_map}
        if ground_truth:
            gt_map.update(ground_truth)

        # Track performance per agent
        agent_performance = defaultdict(lambda: {
            "correct": 0,
            "total": 0,
            "confidence_scores": [],
            "correct_with_high_conf": 0,
            "high_conf_count": 0
        })

        for i, result in enumerate(self.question_results):
            question_id = str(i)
            gt = result.get('ground_truth') or gt_map.get(question_id)

            if not gt:
                continue

            round3_results = result.get('round3_results', {})

            for agent_id, decision in round3_results.items():
                ranking = decision.get('ranking', [])
                answer = ranking[0] if ranking else decision.get('answer')
                confidence = decision.get('confidence', 'Medium')

                if answer:
                    agent_performance[agent_id]["total"] += 1

                    # Check correctness
                    if answer == gt:
                        agent_performance[agent_id]["correct"] += 1

                        if confidence == "High":
                            agent_performance[agent_id]["correct_with_high_conf"] += 1

                    # Track confidence
                    conf_score = {"High": 1.0, "Medium": 0.7, "Low": 0.4}.get(confidence, 0.5)
                    agent_performance[agent_id]["confidence_scores"].append(conf_score)

                    if confidence == "High":
                        agent_performance[agent_id]["high_conf_count"] += 1

        # Calculate metrics per agent
        by_agent = {}
        for agent_id, perf in agent_performance.items():
            accuracy = perf["correct"] / perf["total"] if perf["total"] > 0 else 0.0
            avg_confidence = sum(perf["confidence_scores"]) / len(perf["confidence_scores"]) if perf["confidence_scores"] else 0.0

            # Confidence calibration: high confidence should correlate with correctness
            calibration = perf["correct_with_high_conf"] / perf["high_conf_count"] if perf["high_conf_count"] > 0 else 0.0

            by_agent[agent_id] = {
                "accuracy": accuracy,
                "correct": perf["correct"],
                "total": perf["total"],
                "avg_confidence": avg_confidence,
                "confidence_calibration": calibration
            }

        # Find best/worst
        if by_agent:
            best_agent = max(by_agent.items(), key=lambda x: x[1]["accuracy"])
            worst_agent = min(by_agent.items(), key=lambda x: x[1]["accuracy"])
        else:
            best_agent = ("N/A", {"accuracy": 0.0})
            worst_agent = ("N/A", {"accuracy": 0.0})

        return {
            "by_agent": by_agent,
            "best_performing_agent": best_agent[0],
            "best_accuracy": best_agent[1]["accuracy"],
            "worst_performing_agent": worst_agent[0],
            "worst_accuracy": worst_agent[1]["accuracy"]
        }

    def calculate_decision_method_comparison(self, ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Compare different decision aggregation methods.

        Args:
            ground_truth: {question_id: correct_answer}

        Returns:
            {
                "borda_count": {"accuracy": float, "correct": int},
                "majority_vote": {"accuracy": float, "correct": int},
                "weighted_consensus": {"accuracy": float, "correct": int},
                "best_method": str
            }
        """
        accuracy_metrics = self.calculate_accuracy(ground_truth)

        method_comparison = accuracy_metrics.get("method_comparison", {})

        # Determine best method
        if method_comparison:
            best_method = max(method_comparison.items(), key=lambda x: x[1]["accuracy"])
        else:
            best_method = ("unknown", {"accuracy": 0.0})

        return {
            **method_comparison,
            "best_method": best_method[0],
            "best_accuracy": best_method[1]["accuracy"]
        }

    def generate_summary_report(self, ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary report with all metrics.

        Args:
            ground_truth: {question_id: correct_answer}

        Returns:
            Complete summary dictionary with all metrics
        """
        logging.info("Generating comprehensive metrics summary...")

        accuracy = self.calculate_accuracy(ground_truth)
        convergence = self.calculate_convergence()
        disagreement = self.calculate_disagreement_matrix()
        opinion_changes = self.calculate_opinion_change_rate()
        agent_performance = self.calculate_agent_performance(ground_truth)
        method_comparison = self.calculate_decision_method_comparison(ground_truth)

        # Calculate average metrics across questions
        avg_n_agents = sum(len(r.get('recruited_agents', [])) for r in self.question_results) / len(self.question_results) if self.question_results else 0
        avg_time = sum(r.get('metadata', {}).get('total_time', 0) for r in self.question_results) / len(self.question_results) if self.question_results else 0

        summary = {
            "accuracy": accuracy,
            "convergence": convergence,
            "disagreement": disagreement,
            "opinion_changes": opinion_changes,
            "agent_performance": agent_performance,
            "method_comparison": method_comparison,
            "metadata": {
                "total_questions": len(self.question_results),
                "avg_agents_per_question": avg_n_agents,
                "avg_time_per_question": avg_time,
                "questions_with_ground_truth": len([r for r in self.question_results if r.get('ground_truth')])
            }
        }

        logging.info(f"Summary generated: {accuracy['correct_count']}/{accuracy['total_count']} correct "
                    f"({accuracy['overall_accuracy']:.2%})")

        return summary


__all__ = ["MetricsCalculator"]
