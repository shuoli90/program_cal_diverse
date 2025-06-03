# Quality prompts

SYSTEM_CREATIVE_QUALITY_PROMPT = (
    """You are an impartial judge evaluating the quality of creative writing responses by an AI assistant, following a detailed rubric."""
)
USER_CREATIVE_QUALITY_PROMPT = """Please help me evaluate the quality of the creative writing response provided by an AI assistant to the user's prompt below, based on the detailed rubric provided.

Consider the following rubric criteria while evaluating:
1. Overall/holistic/cohesive readability of the story (not just a compilation of elements).
2. Use of key narrative elements - vocabulary choice, imagery, setting, themes, dialogue, characterisation, point of view.
3. Structural elements and presentation which reflect control of structural elements such as spelling, grammar, punctuation, paragraphing, and formatting.
4. Overall plot logic: hook, conflict, initial crisis, rising and falling action, denouement/resolution (Freitag’s pyramid).
5. Creativity/innovation/originality/research—credibility, new knowledge, avoidance of cliché and derivative tropes.
6. Incorporation of the John Kennedy Toole style of writing using the indicators/characteristics listed.
7. Understanding and habitation of the epic genre of heroic/legendary adventure.
8. Description and credibility of a single combat scene.
9. Accurate inclusion of two main characters Ignatius J. Reilly and a pterodactyl in action and description.
10. Use of a characteristically dark humorous tone.

Each criterion is scored out of 10 points. Use the following marking guideline:
- Emerging: 1-4
- Competent: 5-8
- Sophisticated: 9-10

Provide a total score out of 100 points by summing your scores for each criterion.

[User Creative Writing Prompt]
{question}
[The Start of Assistant's Response]
{answer}
[The End of Assistant's Response]

After providing the total numerical score, give a clear and unbiased explanation of your evaluation for each criterion.
"""

SYSTEM_ARGUMENTATIVE_QUALITY_PROMPT = (
"""You are an impartial judge evaluating the quality of argumentative writing responses provided by an AI assistant."""
)
USER_ARGUMENTATIVE_QUALITY_PROMPT = """Please help me evaluate the quality of the argumentative writing response provided by an AI assistant to the user's prompt below, based on the criteria provided.

Consider the following criteria while evaluating:
1. Clarity and Accuracy: Is the text free from grammatical, punctuation, and spelling errors? Is the writing clear, coherent, and understandable?
2. Thesis and Argument Strength: Does the essay clearly state a thesis or main argument? How strong and well-articulated are the supporting arguments?
3. Evidence and Credibility: Is the supporting evidence credible, relevant, and effectively integrated to strengthen the arguments?
4. Logical Structure and Flow: Do arguments and evidence logically build upon each other? Are transitions smooth, and is the overall structure coherent?
5. Counterarguments and Refutations: Are counterarguments adequately identified, addressed, and effectively refuted or acknowledged?
6. Depth and Complexity: Does the essay demonstrate a nuanced understanding and thoughtful analysis of the issue's complexity?
7. Originality and Perspective: Does the essay present a fresh perspective, avoiding clichés and simplistic arguments?
8. Ethical and Responsible Writing: Does the essay responsibly handle sensitive issues, avoiding harmful, toxic, or offensive perspectives?

Each criterion is scored out of 10 points. Use the following marking guideline:
- Emerging: 1-4
- Competent: 5-8
- Sophisticated: 9-10

Provide a total score out of 80 points by summing your scores for each criterion.

[User Argumentative Writing Prompt]
{question}
[The Start of Assistant's Response]
{answer}
[The End of Assistant's Response]

After providing the total numerical score, give a clear and unbiased explanation of your evaluation for each criterion.
"""


# Diversity prompts

SYSTEM_CREATIVE_DIVERSITY_PROMPT = (
    """You are an impartial judge evaluating the similarity between responses provided by two AI assistants in creative writing, including semantic, lexical, and syntactical aspects."""
)
USER_CREATIVE_DIVERSITY_PROMPT = """Please help me evaluate the similarity between the responses provided by two AI assistants to the user's creative writing prompt below. Your goal is to assign a single score that measures how similar the two responses are, focusing on semantic overlap as well as lexical and syntactical similarity.

Consider the following criteria while evaluating similarity:
1. Semantic Overlap: Do the responses share similar underlying themes, ideas, narrative elements, or emotional content?
2. Lexical Similarity: Are similar words or phrases frequently used across the two responses?
3. Syntactical Similarity: Do the responses exhibit similar sentence structures or stylistic patterns?
4. Thematic Consistency: Do both responses explore similar themes or motifs?
5. Stylistic Harmony: Are the storytelling approaches, tones, and literary styles closely aligned?

Provide a single score on a scale from 0 to 10. A highly similar pair of responses should receive a score above 5, whereas a less similar pair should score below 5.

[User Creative Writing Prompt]
{question}
[The Start of Assistant A's Response]
{answer1}
[The End of Assistant A's Response]
[The Start of Assistant B's Response]
{answer2}
[The End of Assistant B's Response]

As you evaluate, maintain objectivity and avoid biases related to response length or presentation order. First, output a single numerical score indicating the similarity of the two responses. In the next line, provide a clear and unbiased explanation of your evaluation.
"""

SYSTEM_ARGUMENTATIVE_DIVERSITY_PROMPT = (
    """You are an impartial judge evaluating the similarity between responses provided by two AI assistants in argumentative writing, including semantic, lexical, and syntactical aspects."""
)
USER_ARGUMENTATIVE_DIVERSITY_PROMPT = """Please help me evaluate the similarity between the responses provided by two AI assistants to the user's argumentative writing prompt below. Your goal is to assign a single score that measures how similar the two responses are, focusing on semantic overlap, lexical choice, and syntactical structures.

Consider the following criteria while evaluating similarity:

1. Semantic Overlap: Do the responses present similar arguments, reasoning, or viewpoints?
2. Lexical Similarity: Are similar terms, phrases, or key argumentative language frequently used across the two responses?
3. Syntactical Similarity: Do the responses exhibit similar argumentative structures, sentence patterns, or rhetorical devices?
4. Argumentative Consistency: Do both responses consistently support or oppose similar positions?
5. Rhetorical Style: Are the persuasive strategies, tones, and rhetorical approaches closely aligned?

Provide a single score on a scale from 0 to 10. A highly similar pair of argumentative responses should receive a score above 5, whereas a less similar pair should score below 5.

[User Argumentative Writing Prompt]
{question}
[The Start of Assistant A's Response]
{answer1}
[The End of Assistant A's Response]
[The Start of Assistant B's Response]
{answer2}
[The End of Assistant B's Response]

As you evaluate, maintain objectivity and avoid biases related to response length or presentation order. First, output a single numerical score indicating the similarity of the two responses. In the next line, provide a clear and unbiased explanation of your evaluation.
"""