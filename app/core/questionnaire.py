from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Answer:
    text: str
    color: str
    value: int


@dataclass
class Question:
    text: str
    possible_answers: dict[str, Answer]


@dataclass
class Questionnaire:
    questions: list[Question]
    responses: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Questionnaire:
        with path.open() as f:
            data = yaml.safe_load(f)

        questions = [
            Question(
                text=question["question"],
                possible_answers={
                    key: Answer(
                        text=answer["answer"],
                        color=answer["color"],
                        value=answer["answer_value"],
                    )
                    for key, answer in question["possible_answers"].items()
                },
            )
            for question in data["questions"]
        ]
        return cls(questions=questions)

    def set_response(self, question_text: str, answer_key: str) -> None:
        self.responses[question_text] = answer_key

    def save(self, path: Path) -> None:
        """Save the user's answers (as recorded via set_response) to a YAML file."""
        data = {
            question.text: {
                "answer_key": answer_key,
                "answer": question.possible_answers[answer_key].text,
                "answer_value": question.possible_answers[answer_key].value,
            }
            for question in self.questions
            if (answer_key := self.responses.get(question.text)) is not None
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(
                data, f, sort_keys=False, allow_unicode=True, default_flow_style=False
            )
