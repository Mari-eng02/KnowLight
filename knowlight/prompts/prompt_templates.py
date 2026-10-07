"""
Prompt Templates for Recommendation Graph Instruction Tuning and Online Inference in KnowLight.
"""


class PromptTemplates:

    @staticmethod
    def build_prompt(
        user_id: str,
        neighbors_1: str,
        neighbors_2: str,
        candidate_item: str
    ) -> str:
        """
        Prompt (Score Distillation):
        Returns the structural context for the user and the specific candidate item to evaluate.
        Instruction and Query are injected during fine-tuning.
        """
        structural_desc = (
            f"The target user (ID: {user_id}) is embedded in a multi-hop recommendation graph.\n"
            f"At Hop 0, the target user entity is identified as User {user_id}.\n"
            f"At Hop 1, the target user is directly connected to the following item entities in the graph: {neighbors_1}.\n"
            f"At Hop 2, these items connect the target user to neighboring co-interacted users and preference communities: {neighbors_2}.\n"
            "The multi-hop graph captures collaborative filtering signals, semantic similarity, and user preference communities.\n\n"
            f"Candidate Item to evaluate: {candidate_item}"
        )
        return structural_desc

    