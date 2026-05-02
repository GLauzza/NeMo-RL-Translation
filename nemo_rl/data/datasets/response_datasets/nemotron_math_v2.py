from typing import Any, Optional

from datasets import Dataset, load_dataset, load_from_disk

from nemo_rl.data.interfaces import TaskDataSpec


def format_math(
    data: dict[str, str | float | int], output_key: str = "answer"
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["question_fr"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def prepare_nemotron_math_v2_dataset(
    split: str = "low",
    seed: int = 42,
    test_size: float = 0.05,
    output_key: str = "answer",
) -> dict[str, Dataset | None]:
    """Load and split the Nemotron-Math-v2 dataset into train and validation sets using HF's train_test_split."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    original_ds = load_from_disk("/lustre/fsn1/projects/rech/knb/ukq43aj/Datasets/Nemotron-Math-v2-FR-answer")
    original_ds = original_ds.shuffle().filter(lambda x: len(x["tools"]) == 0)

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(
        format_math,
        remove_columns=split_ds["train"].column_names,
        fn_kwargs={"output_key": output_key},
    )
    val_formatted = split_ds["test"].map(
        format_math,
        remove_columns=split_ds["test"].column_names,
        fn_kwargs={"output_key": output_key},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class Nemotron_math_v2_Dataset:
    def __init__(
        self,
        split: str = "low",
        seed: int = 42,
        test_size: float = 0.05,
        output_key: str = "answer",
        prompt_file: Optional[str] = None,
    ):
        """Initialize the Nemotron_math_v2 dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        # train, train_1M, train_2M, and train_5M are supported splits.
        if split not in ["high_part00", "high_part01", "high_part02", "medium", "low"]:
            raise ValueError(
                f'Invalid split: {split}. Please use "high_part00", "high_part01", "high_part02", "medium", "low".'
            )

        self.formatted_ds = prepare_nemotron_math_v2_dataset(
            split=split, seed=seed, test_size=test_size, output_key=output_key
        )

        self.task_spec = TaskDataSpec(
            task_name="Nemotron-Math-v2",
            prompt_file=prompt_file,
        )
