from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omnivault.dsa.algorithms.strings import (
    SuffixArray,
    aho_corasick_search,
    all_palindromic_substrings,
    boyer_moore_search,
    edit_distance,
    kmp_search,
    longest_common_subsequence,
    longest_common_substring,
    longest_palindromic_substring,
    manacher_palindromes,
    rabin_karp_search,
    z_algorithm,
    z_search,
)

if TYPE_CHECKING:
    pass


class TestStringSearchAlgorithms:
    @pytest.fixture
    def text_and_pattern(self) -> tuple[str, str]:
        return "ABABDABACDABABCABCABCABCABC", "ABABCABCABCABC"

    @pytest.mark.unit
    @pytest.mark.parametrize("search_func", [kmp_search, rabin_karp_search, z_search, boyer_moore_search])
    def test_string_search_algorithms_find_pattern(
        self, search_func: object, text_and_pattern: tuple[str, str]
    ) -> None:
        text, pattern = text_and_pattern

        matches = search_func(text, pattern)  # type: ignore

        assert isinstance(matches, list)
        assert len(matches) > 0
        for match in matches:
            assert text[match : match + len(pattern)] == pattern

    @pytest.mark.unit
    def test_kmp_search_multiple_matches(self) -> None:
        text = "ABABABABAB"
        pattern = "ABAB"

        matches = kmp_search(text, pattern)

        assert matches == [0, 2, 4, 6]

    @pytest.mark.unit
    def test_kmp_search_no_matches(self) -> None:
        text = "ABCDEFG"
        pattern = "XYZ"

        matches = kmp_search(text, pattern)

        assert matches == []

    @pytest.mark.unit
    def test_kmp_search_empty_pattern(self) -> None:
        text = "ABCDEFG"
        pattern = ""

        matches = kmp_search(text, pattern)

        assert matches == []

    @pytest.mark.unit
    def test_rabin_karp_search_with_collisions(self) -> None:
        text = "ABCABCABC"
        pattern = "ABC"

        matches = rabin_karp_search(text, pattern, prime=7)

        assert matches == [0, 3, 6]

    @pytest.mark.unit
    def test_boyer_moore_search_efficient_skipping(self) -> None:
        text = "ABAAABCDABCDABDE"
        pattern = "ABCD"

        matches = boyer_moore_search(text, pattern)

        assert matches == [4, 8]

    @pytest.mark.unit
    def test_z_algorithm_basic(self) -> None:
        s = "aabaaab"

        z_values = z_algorithm(s)

        assert z_values[0] == 7
        assert z_values[1] == 1
        assert z_values[2] == 0
        assert z_values[3] == 2
        assert z_values[4] == 3
        assert z_values[5] == 1
        assert z_values[6] == 0

    @pytest.mark.unit
    def test_z_search_pattern_matching(self) -> None:
        text = "abcabcabcabc"
        pattern = "abc"

        matches = z_search(text, pattern)

        assert matches == [0, 3, 6, 9]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("text", "pattern", "expected_count"),
        [
            ("hello world", "lo", 1),
            ("aaaa", "aa", 3),
            ("abcdef", "xyz", 0),
            ("", "a", 0),
            ("a", "", 0),
        ],
    )
    def test_search_algorithms_parametrized(self, text: str, pattern: str, expected_count: int) -> None:
        kmp_matches = kmp_search(text, pattern)
        rabin_karp_matches = rabin_karp_search(text, pattern)
        z_matches = z_search(text, pattern)
        boyer_moore_matches = boyer_moore_search(text, pattern)

        assert len(kmp_matches) == expected_count
        assert len(rabin_karp_matches) == expected_count
        assert len(z_matches) == expected_count
        assert len(boyer_moore_matches) == expected_count

        if expected_count > 0:
            assert kmp_matches == rabin_karp_matches == z_matches == boyer_moore_matches


class TestPalindromeAlgorithms:
    @pytest.mark.unit
    def test_longest_palindromic_substring(self) -> None:
        s = "babad"

        result = longest_palindromic_substring(s)

        assert result in ["bab", "aba"]

    @pytest.mark.unit
    def test_longest_palindromic_substring_even_length(self) -> None:
        s = "cbbd"

        result = longest_palindromic_substring(s)

        assert result == "bb"

    @pytest.mark.unit
    def test_all_palindromic_substrings(self) -> None:
        s = "abc"

        palindromes = all_palindromic_substrings(s)

        expected = {"a", "b", "c"}
        assert set(palindromes) == expected

    @pytest.mark.unit
    def test_all_palindromic_substrings_with_longer_palindromes(self) -> None:
        s = "aab"

        palindromes = all_palindromic_substrings(s)

        expected = {"a", "aa", "b"}
        assert set(palindromes) == expected

    @pytest.mark.unit
    def test_manacher_palindromes_simple(self) -> None:
        s = "abccba"

        lengths = manacher_palindromes(s)

        assert max(lengths) == 6

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("s", "expected_longest"),
        [
            ("a", "a"),
            ("aa", "aa"),
            ("aba", "aba"),
            ("abcd", "a"),
            ("racecar", "racecar"),
        ],
    )
    def test_palindrome_algorithms_parametrized(self, s: str, expected_longest: str) -> None:
        result = longest_palindromic_substring(s)
        assert len(result) == len(expected_longest)


class TestStringComparisonAlgorithms:
    @pytest.mark.unit
    def test_longest_common_substring(self) -> None:
        str1 = "GeeksforGeeks"
        str2 = "GeeksQuiz"

        result = longest_common_substring(str1, str2)

        assert result == "Geeks"

    @pytest.mark.unit
    def test_longest_common_substring_no_common(self) -> None:
        str1 = "abc"
        str2 = "def"

        result = longest_common_substring(str1, str2)

        assert result == ""

    @pytest.mark.unit
    def test_longest_common_subsequence(self) -> None:
        str1 = "ABCDGH"
        str2 = "AEDFHR"

        result = longest_common_subsequence(str1, str2)

        assert result == "ADH"

    @pytest.mark.unit
    def test_longest_common_subsequence_identical_strings(self) -> None:
        str1 = "ABCD"
        str2 = "ABCD"

        result = longest_common_subsequence(str1, str2)

        assert result == "ABCD"

    @pytest.mark.unit
    def test_edit_distance_basic(self) -> None:
        str1 = "kitten"
        str2 = "sitting"

        distance = edit_distance(str1, str2)

        assert distance == 3

    @pytest.mark.unit
    def test_edit_distance_identical_strings(self) -> None:
        str1 = "hello"
        str2 = "hello"

        distance = edit_distance(str1, str2)

        assert distance == 0

    @pytest.mark.unit
    def test_edit_distance_empty_strings(self) -> None:
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3
        assert edit_distance("", "") == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("str1", "str2", "expected_distance"),
        [
            ("cat", "bat", 1),
            ("saturday", "sunday", 3),
            ("intention", "execution", 5),
            ("abc", "abc", 0),
        ],
    )
    def test_edit_distance_parametrized(self, str1: str, str2: str, expected_distance: int) -> None:
        distance = edit_distance(str1, str2)
        assert distance == expected_distance


class TestSuffixArray:
    @pytest.mark.unit
    def test_suffix_array_construction(self) -> None:
        text = "banana"
        suffix_array = SuffixArray(text)

        assert suffix_array.suffix_array == [5, 3, 1, 0, 4, 2]

    @pytest.mark.unit
    def test_suffix_array_search(self) -> None:
        text = "banana"
        suffix_array = SuffixArray(text)

        matches = suffix_array.search("an")

        assert sorted(matches) == [1, 3]

    @pytest.mark.unit
    def test_suffix_array_search_no_matches(self) -> None:
        text = "banana"
        suffix_array = SuffixArray(text)

        matches = suffix_array.search("xyz")

        assert matches == []

    @pytest.mark.unit
    def test_suffix_array_empty_text(self) -> None:
        text = ""
        suffix_array = SuffixArray(text)

        assert suffix_array.suffix_array == []
        assert suffix_array.lcp_array == []
        assert suffix_array.search("a") == []

    @pytest.mark.unit
    def test_suffix_array_single_character(self) -> None:
        text = "a"
        suffix_array = SuffixArray(text)

        assert suffix_array.suffix_array == [0]
        assert suffix_array.search("a") == [0]


class TestAhoCorasick:
    @pytest.mark.unit
    def test_aho_corasick_multiple_patterns(self) -> None:
        text = "ushers"
        patterns = ["he", "she", "his", "hers"]

        results = aho_corasick_search(text, patterns)

        assert results["he"] == [2]
        assert results["she"] == [1]
        assert results["his"] == []
        assert results["hers"] == [2]

    @pytest.mark.unit
    def test_aho_corasick_overlapping_patterns(self) -> None:
        text = "abcdefg"
        patterns = ["abc", "bcd", "cde"]

        results = aho_corasick_search(text, patterns)

        assert results["abc"] == [0]
        assert results["bcd"] == [1]
        assert results["cde"] == [2]

    @pytest.mark.unit
    def test_aho_corasick_no_patterns(self) -> None:
        text = "hello world"
        patterns: list[str] = []

        results = aho_corasick_search(text, patterns)

        assert results == {}

    @pytest.mark.unit
    def test_aho_corasick_pattern_not_found(self) -> None:
        text = "hello world"
        patterns = ["xyz", "abc"]

        results = aho_corasick_search(text, patterns)

        assert results["xyz"] == []
        assert results["abc"] == []

    @pytest.mark.unit
    def test_aho_corasick_repeated_patterns(self) -> None:
        text = "ababab"
        patterns = ["ab", "ba"]

        results = aho_corasick_search(text, patterns)

        assert results["ab"] == [0, 2, 4]
        assert results["ba"] == [1, 3]


class TestStringAlgorithmEdgeCases:
    @pytest.mark.unit
    def test_algorithms_with_unicode(self) -> None:
        text = "héllo wörld"
        pattern = "ör"

        kmp_matches = kmp_search(text, pattern)
        rabin_karp_matches = rabin_karp_search(text, pattern)

        assert kmp_matches == [7]
        assert rabin_karp_matches == [7]

    @pytest.mark.unit
    def test_algorithms_with_special_characters(self) -> None:
        text = "hello!@#$%world"
        pattern = "!@#"

        matches = kmp_search(text, pattern)

        assert matches == [5]

    @pytest.mark.unit
    def test_very_long_pattern(self) -> None:
        text = "short"
        pattern = "very_long_pattern_that_exceeds_text_length"

        matches = kmp_search(text, pattern)

        assert matches == []
