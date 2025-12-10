"""
Case Study Extractor - Comprehensive Analysis of Multi-Agent Medical Reasoning

Extracts detailed interaction logs from a single question run with all teamwork
components activated, formatting them into a comprehensive case study narrative.

Usage:
    python extract_case_study.py --result-file path/to/q001_results.json --output case_study.md

    # Or extract from a run directory
    python extract_case_study.py --run-dir multi-agent-gemma/results/medqa_1q_all_active_run1 --question-id q001
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def format_agent_list(agents: list) -> str:
    """Format recruited agents list."""
    output = []
    for agent in agents:
        output.append(f"- **{agent['agent_id']}**: {agent['role']}")
        output.append(f"  - Expertise: {agent['expertise']}")
    return "\n".join(output)


def format_round_results(round_data: Dict, round_num: int) -> str:
    """Format agent responses from a round."""
    output = [f"\n### Round {round_num} Responses\n"]

    for agent_id, response in round_data.items():
        output.append(f"#### {agent_id}")
        output.append("```")
        if isinstance(response, dict):
            # Round 3 format
            output.append(f"Answer: {response.get('answer', 'N/A')}")
            output.append(f"Ranking: {response.get('ranking', [])}")
            output.append(f"Confidence: {response.get('confidence', 'N/A')}")
            output.append(f"\nFull Response:\n{response.get('raw', '')}")
        else:
            # Round 1 & 2 format
            output.append(response)
        output.append("```\n")

    return "\n".join(output)


def format_teamwork_interactions(teamwork: Dict) -> str:
    """Format teamwork component interactions."""
    if not teamwork:
        return "\n*No teamwork interactions recorded*\n"

    output = ["\n## Teamwork Component Interactions\n"]

    # Shared Mental Model (SMM)
    if 'smm' in teamwork:
        smm = teamwork['smm']
        output.append("### Shared Mental Model (SMM)")
        output.append("\n**Question Analysis:**")
        output.append("```")
        output.append(smm.get('question_analysis', 'N/A'))
        output.append("```")

        if smm.get('verified_facts'):
            output.append("\n**Verified Facts:**")
            for fact in smm['verified_facts']:
                output.append(f"- {fact}")

        if smm.get('debated_points'):
            output.append("\n**Debated Points:**")
            for point in smm['debated_points']:
                output.append(f"- {point}")
        output.append("")

    # Leadership
    if 'leadership' in teamwork:
        lead = teamwork['leadership']
        output.append("### Leadership")

        if 'initial_direction' in lead:
            output.append("\n**Initial Direction:**")
            output.append("```")
            output.append(lead['initial_direction'])
            output.append("```")

        if 'corrections' in lead and lead['corrections']:
            output.append("\n**Leadership Corrections:**")
            for correction in lead['corrections']:
                output.append(f"\n**Turn {correction.get('turn', '?')}:**")
                output.append("```")
                output.append(correction.get('correction', 'N/A'))
                output.append("```")
        output.append("")

    # Trust Network
    if 'trust' in teamwork:
        trust = teamwork['trust']
        output.append("### Trust Network")

        if 'initial_scores' in trust:
            output.append("\n**Initial Trust Scores:**")
            for agent_id, score in trust['initial_scores'].items():
                output.append(f"- {agent_id}: {score:.2f}")

        if 'final_scores' in trust:
            output.append("\n**Final Trust Scores:**")
            for agent_id, score in trust['final_scores'].items():
                output.append(f"- {agent_id}: {score:.2f}")

        if 'adjustments' in trust and trust['adjustments']:
            output.append("\n**Trust Adjustments:**")
            for adj in trust['adjustments']:
                output.append(f"- {adj.get('agent', '?')}: {adj.get('reason', 'N/A')}")
        output.append("")

    # Team Orientation
    if 'team_orientation' in teamwork:
        to = teamwork['team_orientation']
        output.append("### Team Orientation")

        if 'role_assignments' in to:
            output.append("\n**Role-Based Weights:**")
            for role_info in to['role_assignments']:
                output.append(f"- **{role_info['agent_id']}** ({role_info['role']}): Weight = {role_info['weight']:.2f}")

        if 'formal_report' in to:
            output.append("\n**Formal Team Report:**")
            output.append("```")
            output.append(to['formal_report'])
            output.append("```")
        output.append("")

    # Mutual Monitoring
    if 'mutual_monitoring' in teamwork:
        mm = teamwork['mutual_monitoring']
        output.append("### Mutual Monitoring")

        if 'challenges' in mm and mm['challenges']:
            output.append("\n**Peer Challenges:**")
            for challenge in mm['challenges']:
                output.append(f"\n**Turn {challenge.get('turn', '?')}:**")
                output.append(f"- Target: {challenge.get('target_agent', '?')}")
                output.append(f"- Challenger: {challenge.get('challenger', '?')}")
                output.append("```")
                output.append(challenge.get('challenge', 'N/A'))
                output.append("```")
                if 'defense' in challenge:
                    output.append("**Defense:**")
                    output.append("```")
                    output.append(challenge['defense'])
                    output.append("```")

        if 'validation_passed' in mm:
            output.append(f"\n**Validation Status:** {'[PASSED]' if mm['validation_passed'] else '[FAILED]'}")
        output.append("")

    # Post-Round 2 Processing
    if 'post_r2_processing' in teamwork:
        output.append("### Post-Round 2 Processing")
        post = teamwork['post_r2_processing']
        if post.get('formal_report_created'):
            output.append("- Formal team report created")
        if post.get('trust_adjusted'):
            output.append("- Trust scores adjusted")
        if post.get('monitoring_performed'):
            output.append("- Mutual monitoring performed")
        output.append("")

    return "\n".join(output)


def format_final_decision(decision: Dict) -> str:
    """Format final decision and aggregation."""
    output = ["\n## Final Decision\n"]

    output.append(f"**Selected Answer:** {decision.get('primary_answer', 'N/A')}")

    borda = decision.get('borda_count', {})
    if borda:
        output.append(f"\n**Aggregation Method:** {borda.get('method', 'N/A')}")
        output.append(f"**Agreement Rate:** {borda.get('agreement_rate', 0):.1%}")

        output.append("\n**Borda Count Scores:**")
        scores = borda.get('scores', {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for option, score in sorted_scores:
            output.append(f"- {option}: {score:.2f}")

    convergence = decision.get('convergence', {})
    if convergence:
        output.append(f"\n**Convergence:**")
        output.append(f"- Converged: {convergence.get('converged', False)}")
        output.append(f"- First Choice Agreement: {convergence.get('first_choice_agreement', 0):.1%}")
        output.append(f"- Ranking Similarity: {convergence.get('ranking_similarity', 0):.2f}")

    return "\n".join(output)


def format_metadata(metadata: Dict, token_usage: Dict) -> str:
    """Format metadata and performance metrics."""
    output = ["\n## Performance Metrics\n"]

    # Timing
    output.append("### Timing Breakdown")
    output.append(f"- **Total Time:** {metadata.get('total_time', 0):.2f}s")
    output.append(f"- **Recruitment:** {metadata.get('recruit_time', 0):.2f}s")
    output.append(f"- **Round 1:** {metadata.get('round1_time', 0):.2f}s")
    output.append(f"- **Round 2:** {metadata.get('round2_time', 0):.2f}s")
    output.append(f"- **Round 3:** {metadata.get('round3_time', 0):.2f}s")
    output.append(f"- **Aggregation:** {metadata.get('aggregation_time', 0):.4f}s")

    # API Calls
    output.append(f"\n### API Usage")
    output.append(f"- **Total API Calls:** {metadata.get('api_calls', 0)}")
    output.append(f"- **Number of Agents:** {metadata.get('n_agents', 0)}")

    # Tokens
    output.append(f"\n### Token Usage")
    output.append(f"- **Input Tokens:** {token_usage.get('input_tokens', 0):,}")
    output.append(f"- **Output Tokens:** {token_usage.get('output_tokens', 0):,}")
    output.append(f"- **Total Tokens:** {token_usage.get('total_tokens', 0):,}")
    if token_usage.get('has_image'):
        output.append(f"- **Image Tokens:** {token_usage.get('image_tokens', 0):,}")

    return "\n".join(output)


def create_case_study(result_data: Dict) -> str:
    """Generate comprehensive case study from result data."""

    # Header
    output = [
        "# Multi-Agent Medical Reasoning Case Study",
        f"\n**Question ID:** {result_data['question_id']}",
        f"**Date:** {result_data['metadata']['timestamp']}",
        f"**Framework:** {result_data['metadata']['framework']}",
        "\n---\n"
    ]

    # Question
    output.append("## Clinical Question\n")
    output.append(result_data['question'])

    # Options
    if result_data.get('options'):
        output.append("\n**Options:**")
        for option in result_data['options']:
            output.append(f"- {option}")

    # Ground Truth
    output.append(f"\n**Ground Truth Answer:** {result_data.get('ground_truth', 'N/A')}")
    output.append(f"**System Answer:** {result_data['final_decision']['primary_answer']}")
    is_correct = result_data.get('is_correct', False)
    output.append(f"**Result:** {'[CORRECT]' if is_correct else '[INCORRECT]'}")

    # Recruited Agents
    output.append("\n---\n\n## Recruited Agents\n")
    output.append(format_agent_list(result_data.get('recruited_agents', [])))

    # Round 1
    output.append("\n---\n")
    output.append(format_round_results(result_data.get('round1_results', {}), 1))

    # Round 2
    output.append("\n---\n")
    output.append(format_round_results(result_data.get('round2_results', {}), 2))

    # Round 3
    output.append("\n---\n")
    output.append(format_round_results(result_data.get('round3_results', {}), 3))

    # Teamwork Interactions
    output.append("\n---\n")
    output.append(format_teamwork_interactions(result_data.get('teamwork_interactions', {})))

    # Final Decision
    output.append("\n---\n")
    output.append(format_final_decision(result_data.get('final_decision', {})))

    # Metadata
    output.append("\n---\n")
    output.append(format_metadata(
        result_data.get('metadata', {}),
        result_data.get('token_usage', {})
    ))

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract comprehensive case study from multi-agent results"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--result-file', type=str,
                      help='Path to question result JSON file (e.g., q001_results.json)')
    group.add_argument('--run-dir', type=str,
                      help='Path to run directory (will extract from questions/ subdirectory)')

    parser.add_argument('--question-id', type=str, default='q001',
                       help='Question ID to extract (default: q001), used with --run-dir')
    parser.add_argument('--output', type=str, default='case_study.md',
                       help='Output markdown file (default: case_study.md)')

    args = parser.parse_args()

    # Determine result file path
    if args.result_file:
        result_path = Path(args.result_file)
    else:
        run_dir = Path(args.run_dir)
        result_path = run_dir / 'questions' / f'{args.question_id}_results.json'

    # Load result data
    if not result_path.exists():
        print(f"Error: Result file not found: {result_path}")
        return 1

    print(f"Loading result data from: {result_path}")
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    # Generate case study
    print("Generating case study...")
    case_study = create_case_study(result_data)

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(case_study)

    print(f"\n[SUCCESS] Case study saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Question: {result_data['question'][:80]}...")
    print(f"  Agents: {len(result_data.get('recruited_agents', []))}")
    print(f"  Ground Truth: {result_data.get('ground_truth', 'N/A')}")
    print(f"  System Answer: {result_data['final_decision']['primary_answer']}")
    print(f"  Correct: {'Yes' if result_data.get('is_correct', False) else 'No'}")
    print(f"  Total Time: {result_data['metadata'].get('total_time', 0):.2f}s")
    print(f"  API Calls: {result_data['metadata'].get('api_calls', 0)}")

    # List teamwork components used
    teamwork = result_data.get('teamwork_interactions', {})
    if teamwork:
        print(f"\n  Teamwork Components:")
        for component in teamwork.keys():
            print(f"    - {component}")

    return 0


if __name__ == '__main__':
    exit(main())
