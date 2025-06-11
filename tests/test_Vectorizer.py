import pytest

from pkg.data.utils import Vectorizer


@pytest.fixture
def vectorizer() -> Vectorizer:
    vectorizer = Vectorizer()
    vectorizer._load(
        {
            vectorizer.GROUPS_KEYWORD: ({"b": 0, "a": 1}, {0: "b", 1: "a"}),
            vectorizer.VALUES_KEYWORD: (
                {
                    ("b", 12): 0,
                    ("b", 7): 1,
                    ("a", 1): 2,
                    ("b", 14): 3,
                    ("b", 5): 4,
                    ("a", 7): 5,
                    ("a", 14): 6,
                },
                {
                    0: ("b", 12),
                    1: ("b", 7),
                    2: ("a", 1),
                    3: ("b", 14),
                    4: ("b", 5),
                    5: ("a", 7),
                    6: ("a", 14),
                },
            ),
        }
    )
    return vectorizer


def test_get_groups(vectorizer):
    assert vectorizer.get_groups() == ["a", "b"]


def test_get_num_groups(vectorizer):
    assert vectorizer.get_num_groups() == 2


def test_get_size(vectorizer):
    assert vectorizer.get_size(group="a") == 3
    assert vectorizer.get_size(group="b") == 4
    assert vectorizer.get_size() == 7


def test_encode_value(vectorizer):
    assert vectorizer.encode_value(group="a", value=7) == 5
    assert vectorizer.encode_value(group="a", value=99) == 7

    assert vectorizer.encode_value(group="c", value=1) == 8
    assert vectorizer.encode_value(group="c", value="1") == 9
    assert vectorizer.get_size(group="c") == 2

    vectorizer.freeze()
    assert vectorizer.encode_value(group="a", value=99) == 7
    with pytest.raises(Exception):
        vectorizer.encode_value(group="c", value=123)


def test_encode_group(vectorizer):
    assert vectorizer.encode_group("a") == 1
    assert vectorizer.encode_group("b") == 0
    assert vectorizer.encode_group("x") == 2


def test_decode_value(vectorizer):
    assert vectorizer.decode_value(value=1) == ("b", 7)


def test_decode_group(vectorizer):
    assert vectorizer.decode_group(value=1) == "a"
