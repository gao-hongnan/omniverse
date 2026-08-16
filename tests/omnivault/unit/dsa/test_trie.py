from __future__ import annotations

import pytest

from omnivault.dsa.trees.trie import Trie


class TestTrie:
    @pytest.fixture
    def empty_trie(self) -> Trie[str]:
        return Trie[str]()

    @pytest.fixture
    def sample_trie(self) -> Trie[str]:
        trie = Trie[str]()
        words = ["cat", "car", "card", "care", "careful", "cars", "carry"]
        for word in words:
            trie.insert(word, f"value_{word}")
        return trie

    @pytest.fixture
    def simple_trie(self) -> Trie[None]:
        trie = Trie[None]()
        words = ["the", "a", "there", "answer", "any", "by", "bye", "their"]
        for word in words:
            trie.insert(word)
        return trie

    def test_empty_trie_operations(self, empty_trie: Trie[str]) -> None:
        assert empty_trie.is_empty()
        assert len(empty_trie) == 0
        assert empty_trie.size() == 0
        assert not empty_trie

    def test_empty_trie_errors(self, empty_trie: Trie[str]) -> None:
        with pytest.raises(KeyError):
            empty_trie.search("missing")

        with pytest.raises(KeyError):
            empty_trie.delete("missing")

        with pytest.raises(ValueError):
            empty_trie.insert("")

    def test_single_word_operations(self, empty_trie: Trie[str]) -> None:
        empty_trie.insert("hello", "world")

        assert not empty_trie.is_empty()
        assert len(empty_trie) == 1
        assert empty_trie.size() == 1
        assert bool(empty_trie)

        assert empty_trie.search("hello") == "world"
        assert empty_trie.contains("hello")
        assert "hello" in empty_trie

    def test_insertion_and_search(self, sample_trie: Trie[str]) -> None:
        assert sample_trie.search("cat") == "value_cat"
        assert sample_trie.search("car") == "value_car"
        assert sample_trie.search("card") == "value_card"
        assert sample_trie.search("care") == "value_care"
        assert sample_trie.search("careful") == "value_careful"
        assert sample_trie.search("cars") == "value_cars"
        assert sample_trie.search("carry") == "value_carry"

        assert sample_trie.size() == 7

    def test_insertion_update_existing(self, empty_trie: Trie[str]) -> None:
        empty_trie.insert("test", "first")
        empty_trie.insert("test", "second")

        assert empty_trie.search("test") == "second"
        assert empty_trie.size() == 1

    def test_contains_method(self, sample_trie: Trie[str]) -> None:
        assert sample_trie.contains("cat")
        assert sample_trie.contains("careful")
        assert not sample_trie.contains("dog")
        assert not sample_trie.contains("ca")

    def test_starts_with_prefix(self, sample_trie: Trie[str]) -> None:
        assert sample_trie.starts_with("car")
        assert sample_trie.starts_with("care")
        assert sample_trie.starts_with("c")
        assert not sample_trie.starts_with("dog")
        assert not sample_trie.starts_with("cars123")

    def test_get_words_with_prefix(self, sample_trie: Trie[str]) -> None:
        car_words = list(sample_trie.get_words_with_prefix("car"))
        expected_car_words = ["car", "card", "care", "careful", "cars", "carry"]
        assert sorted(car_words) == sorted(expected_car_words)

        care_words = list(sample_trie.get_words_with_prefix("care"))
        expected_care_words = ["care", "careful"]
        assert sorted(care_words) == sorted(expected_care_words)

        empty_prefix = list(sample_trie.get_words_with_prefix("xyz"))
        assert empty_prefix == []

    def test_autocomplete(self, sample_trie: Trie[str]) -> None:
        completions = sample_trie.autocomplete("car", limit=3)
        assert len(completions) <= 3
        assert all(word.startswith("car") for word in completions)

        all_completions = sample_trie.autocomplete("car", limit=10)
        expected = ["car", "card", "care", "careful", "cars", "carry"]
        assert sorted(all_completions) == sorted(expected)

    def test_deletion_leaf_word(self, sample_trie: Trie[str]) -> None:
        deleted_value = sample_trie.delete("careful")
        assert deleted_value == "value_careful"
        assert not sample_trie.contains("careful")
        assert sample_trie.contains("care")
        assert sample_trie.size() == 6

    def test_deletion_intermediate_word(self, sample_trie: Trie[str]) -> None:
        deleted_value = sample_trie.delete("car")
        assert deleted_value == "value_car"
        assert not sample_trie.contains("car")
        assert sample_trie.contains("card")
        assert sample_trie.contains("care")
        assert sample_trie.size() == 6

    def test_deletion_nonexistent_word(self, sample_trie: Trie[str]) -> None:
        with pytest.raises(KeyError):
            sample_trie.delete("dog")

        with pytest.raises(KeyError):
            sample_trie.delete("ca")

    def test_get_all_words(self, simple_trie: Trie[None]) -> None:
        all_words = list(simple_trie.get_all_words())
        expected_words = ["the", "a", "there", "answer", "any", "by", "bye", "their"]
        assert sorted(all_words) == sorted(expected_words)

    def test_longest_common_prefix(self) -> None:
        trie = Trie[None]()

        # Single word
        trie.insert("test")
        assert trie.longest_common_prefix() == "test"

        trie.clear()

        # Multiple words with common prefix
        words = ["testing", "tester", "test"]
        for word in words:
            trie.insert(word)
        assert trie.longest_common_prefix() == "test"

        trie.clear()

        # No common prefix
        words = ["cat", "dog", "bird"]
        for word in words:
            trie.insert(word)
        assert trie.longest_common_prefix() == ""

    def test_clear_operation(self, sample_trie: Trie[str]) -> None:
        sample_trie.clear()
        assert sample_trie.is_empty()
        assert sample_trie.size() == 0

    def test_iteration_methods(self, sample_trie: Trie[str]) -> None:
        keys = list(sample_trie.keys())
        expected_keys = ["cat", "car", "card", "care", "careful", "cars", "carry"]
        assert sorted(keys) == sorted(expected_keys)

        values = list(sample_trie.values())
        expected_values = [f"value_{word}" for word in expected_keys]
        assert sorted(values) == sorted(expected_values)

        items = list(sample_trie.items())
        expected_items = [(word, f"value_{word}") for word in expected_keys]
        assert sorted(items) == sorted(expected_items)

    def test_magic_methods(self, sample_trie: Trie[str]) -> None:
        assert "cat" in sample_trie
        assert "dog" not in sample_trie
        assert 123 not in sample_trie

        assert sample_trie["cat"] == "value_cat"

        sample_trie["dog"] = "value_dog"
        assert sample_trie.search("dog") == "value_dog"

        del sample_trie["dog"]
        assert not sample_trie.contains("dog")

    def test_iteration(self, simple_trie: Trie[None]) -> None:
        words = list(simple_trie)
        expected_words = ["the", "a", "there", "answer", "any", "by", "bye", "their"]
        assert sorted(words) == sorted(expected_words)

    def test_equality(self) -> None:
        trie1 = Trie[str]()
        trie2 = Trie[str]()

        words = ["test", "testing", "tester"]
        for word in words:
            trie1.insert(word, f"value_{word}")
            trie2.insert(word, f"value_{word}")

        assert trie1 == trie2

        trie2.insert("new", "new_value")
        assert trie1 != trie2

    def test_get_default(self, sample_trie: Trie[str]) -> None:
        assert sample_trie.get_default("cat", "default") == "value_cat"
        assert sample_trie.get_default("dog", "default") == "default"

    def test_pop_operations(self, sample_trie: Trie[str]) -> None:
        value = sample_trie.pop("cat")
        assert value == "value_cat"
        assert not sample_trie.contains("cat")

        with pytest.raises(KeyError):
            sample_trie.pop("dog")

        default_value = sample_trie.pop("dog", "default")
        assert default_value == "default"

    def test_setdefault(self, sample_trie: Trie[str]) -> None:
        value = sample_trie.setdefault("cat", "default")
        assert value == "value_cat"

        value = sample_trie.setdefault("dog", "value_dog")
        assert value == "value_dog"
        assert sample_trie.search("dog") == "value_dog"

    def test_update_operations(self, empty_trie: Trie[str]) -> None:
        other_trie = Trie[str]()
        other_trie.insert("one", "1")
        other_trie.insert("two", "2")

        empty_trie.update(other_trie)
        assert empty_trie.size() == 2
        assert empty_trie.search("one") == "1"
        assert empty_trie.search("two") == "2"

        dict_data = {"three": "3", "four": "4"}
        empty_trie.update(dict_data)
        assert empty_trie.size() == 4

    @pytest.mark.parametrize(
        ("words", "prefix", "expected_count"),
        [
            (["cat", "car", "card"], "ca", 3),
            (["cat", "car", "card"], "car", 2),
            (["cat", "car", "card"], "card", 1),
            (["cat", "car", "card"], "dog", 0),
            (["the", "there", "therm", "they"], "the", 4),
            (["the", "there", "therm", "they"], "ther", 2),
        ],
    )
    def test_prefix_matching_parametrized(self, words: list[str], prefix: str, expected_count: int) -> None:
        trie = Trie[None]()
        for word in words:
            trie.insert(word)

        matches = list(trie.get_words_with_prefix(prefix))
        assert len(matches) == expected_count

    def test_case_sensitivity(self) -> None:
        trie = Trie[str]()
        trie.insert("Test", "uppercase")
        trie.insert("test", "lowercase")

        assert trie.size() == 2
        assert trie.search("Test") == "uppercase"
        assert trie.search("test") == "lowercase"

    def test_unicode_support(self) -> None:
        trie = Trie[str]()
        unicode_words = ["café", "naïve", "résumé", "jalapeño"]

        for word in unicode_words:
            trie.insert(word, f"value_{word}")

        for word in unicode_words:
            assert trie.contains(word)
            assert trie.search(word) == f"value_{word}"

    def test_empty_string_handling(self, empty_trie: Trie[str]) -> None:
        with pytest.raises(ValueError):
            empty_trie.insert("", "empty")

        with pytest.raises(ValueError):
            empty_trie.delete("")

    @pytest.mark.unit
    def test_prefix_matching_returns_all_siblings(self) -> None:
        trie = Trie[None]()
        for word in ["the", "there", "therm", "they"]:
            trie.insert(word)

        matches = set(trie.get_words_with_prefix("ther"))
        assert matches == {"there", "therm"}

    @pytest.mark.unit
    def test_prefix_matching_deep_branching(self) -> None:
        trie = Trie[None]()
        words = ["aab", "aac", "aad", "aae", "aaf"]
        for word in words:
            trie.insert(word)

        matches = set(trie.get_words_with_prefix("aa"))
        assert matches == set(words)

    @pytest.mark.parametrize("size", [100, 1000])
    def test_performance_large_dataset(self, size: int) -> None:
        trie = Trie[int]()

        words = [f"word_{i:04d}" for i in range(size)]

        for i, word in enumerate(words):
            trie.insert(word, i)

        assert trie.size() == size

        for i in range(0, size, 10):
            word = f"word_{i:04d}"
            assert trie.search(word) == i

        prefix_matches = list(trie.get_words_with_prefix("word_00"))
        assert len(prefix_matches) == 100

    def test_memory_efficiency_shared_prefixes(self) -> None:
        trie = Trie[None]()

        common_prefix = "international"
        suffixes = ["", "ize", "ization", "ism", "ist", "istic", "ly"]

        for suffix in suffixes:
            word = common_prefix + suffix
            trie.insert(word)

        assert trie.size() == len(suffixes)

        words_with_prefix = list(trie.get_words_with_prefix(common_prefix))
        assert len(words_with_prefix) == len(suffixes)
