from src.data.datasets import TextLabelDataset


def test_dataset_len_smoke():
    dataset = TextLabelDataset(["a", "b"], [0, 1])
    assert len(dataset) == 2
