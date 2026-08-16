from __future__ import annotations


def kmp_search(text: str, pattern: str) -> list[int]:
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    lps = _compute_lps(pattern)
    matches: list[int] = []

    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches


def _compute_lps(pattern: str) -> list[int]:
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps


def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> list[int]:
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    base = 256
    pattern_hash = 0
    text_hash = 0
    h = 1
    matches: list[int] = []

    for _ in range(m - 1):
        h = (h * base) % prime

    for i in range(m):
        pattern_hash = (base * pattern_hash + ord(pattern[i])) % prime
        text_hash = (base * text_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if pattern_hash == text_hash and text[i : i + m] == pattern:
            matches.append(i)

        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime

    return matches


def z_algorithm(s: str) -> list[int]:
    n = len(s)
    if n == 0:
        return []

    z = [0] * n
    z[0] = n
    left = right = 0

    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > right:
            left, right = i, i + z[i] - 1

    return z


def z_search(text: str, pattern: str) -> list[int]:
    if not pattern:
        return []

    concat = pattern + "$" + text
    z = z_algorithm(concat)

    pattern_len = len(pattern)
    matches: list[int] = [i - pattern_len - 1 for i in range(pattern_len + 1, len(concat)) if z[i] == pattern_len]

    return matches


def boyer_moore_search(text: str, pattern: str) -> list[int]:
    if not pattern:
        return []

    n, m = len(text), len(pattern)
    if m > n:
        return []

    bad_char = _build_bad_char_table(pattern)
    matches: list[int] = []

    shift = 0
    while shift <= n - m:
        j = m - 1

        while j >= 0 and pattern[j] == text[shift + j]:
            j -= 1

        if j < 0:
            matches.append(shift)
            shift += (m - bad_char.get(text[shift + m], -1)) if shift + m < n else 1
        else:
            shift += max(1, j - bad_char.get(text[shift + j], -1))

    return matches


def _build_bad_char_table(pattern: str) -> dict[str, int]:
    return {char: i for i, char in enumerate(pattern)}


def manacher_palindromes(s: str) -> list[int]:
    if not s:
        return []

    processed = "#".join(f"^{s}$")
    n = len(processed)
    palindrome_lengths = [0] * n
    center = right = 0

    for i in range(1, n - 1):
        mirror = 2 * center - i

        if i < right:
            palindrome_lengths[i] = min(right - i, palindrome_lengths[mirror])

        while processed[i + palindrome_lengths[i] + 1] == processed[i - palindrome_lengths[i] - 1]:
            palindrome_lengths[i] += 1

        if i + palindrome_lengths[i] > right:
            center, right = i, i + palindrome_lengths[i]

    return palindrome_lengths


def longest_palindromic_substring(s: str) -> str:
    if not s:
        return ""

    palindrome_lengths = manacher_palindromes(s)
    max_length = max(palindrome_lengths)
    center_index = palindrome_lengths.index(max_length)

    start = (center_index - max_length) // 2
    return s[start : start + max_length]


def all_palindromic_substrings(s: str) -> list[str]:
    if not s:
        return []

    palindromes: list[str] = []
    palindrome_lengths = manacher_palindromes(s)

    for i, length in enumerate(palindrome_lengths):
        if length > 0:
            start = (i - length) // 2
            end = start + length
            palindromes.append(s[start:end])

    return list(set(palindromes))


def longest_common_substring(str1: str, str2: str) -> str:
    m, n = len(str1), len(str2)
    if m == 0 or n == 0:
        return ""

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length = 0
    ending_pos = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > length:
                    length = dp[i][j]
                    ending_pos = i
            else:
                dp[i][j] = 0

    return str1[ending_pos - length : ending_pos]


def longest_common_subsequence(str1: str, str2: str) -> str:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))


def edit_distance(str1: str, str2: str) -> int:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    return dp[m][n]


class SuffixArray:
    def __init__(self, text: str) -> None:
        self.text = text
        self.n = len(text)
        self.suffix_array = self._build_suffix_array()
        self.lcp_array = self._build_lcp_array()

    def _build_suffix_array(self) -> list[int]:
        suffixes = [(self.text[i:], i) for i in range(self.n)]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]

    def _build_lcp_array(self) -> list[int]:
        if self.n == 0:
            return []

        rank = [0] * self.n
        for i, suffix_idx in enumerate(self.suffix_array):
            rank[suffix_idx] = i

        lcp = [0] * (self.n - 1)
        h = 0

        for i in range(self.n):
            if rank[i] == self.n - 1:
                h = 0
                continue

            j = self.suffix_array[rank[i] + 1]
            while i + h < self.n and j + h < self.n and self.text[i + h] == self.text[j + h]:
                h += 1

            lcp[rank[i]] = h
            if h > 0:
                h -= 1

        return lcp

    def search(self, pattern: str) -> list[int]:
        pattern_len = len(pattern)
        if pattern_len == 0:
            return []

        left = self._lower_bound(pattern)
        right = self._upper_bound(pattern)

        matches: list[int] = [self.suffix_array[i] for i in range(left, right)]

        return sorted(matches)

    def _lower_bound(self, pattern: str) -> int:
        left, right = 0, self.n
        while left < right:
            mid = (left + right) // 2
            suffix = self.text[self.suffix_array[mid] :]
            if suffix < pattern:
                left = mid + 1
            else:
                right = mid
        return left

    def _upper_bound(self, pattern: str) -> int:
        left, right = 0, self.n
        while left < right:
            mid = (left + right) // 2
            suffix = self.text[self.suffix_array[mid] :]
            if suffix.startswith(pattern) or suffix < pattern:
                left = mid + 1
            else:
                right = mid
        return left


def aho_corasick_search(text: str, patterns: list[str]) -> dict[str, list[int]]:
    if not patterns:
        return {}

    trie = AhoCorasickTrie()
    for pattern in patterns:
        trie.add_pattern(pattern)

    trie.build_failure_links()
    return trie.search(text)


class AhoCorasickTrie:
    def __init__(self) -> None:
        self.nodes: list[dict[str, int]] = [{}]
        self.outputs: list[list[str]] = [[]]
        self.failure: list[int] = [0]
        self.node_count = 1

    def add_pattern(self, pattern: str) -> None:
        current = 0
        for char in pattern:
            if char not in self.nodes[current]:
                self.nodes[current][char] = self.node_count
                self.nodes.append({})
                self.outputs.append([])
                self.failure.append(0)
                self.node_count += 1
            current = self.nodes[current][char]
        self.outputs[current].append(pattern)

    def build_failure_links(self) -> None:
        queue: list[int] = []

        for node in self.nodes[0].values():
            self.failure[node] = 0
            queue.append(node)

        while queue:
            current = queue.pop(0)

            for char, child in self.nodes[current].items():
                queue.append(child)

                failure = self.failure[current]
                while failure != 0 and char not in self.nodes[failure]:
                    failure = self.failure[failure]

                if char in self.nodes[failure]:
                    self.failure[child] = self.nodes[failure][char]
                else:
                    self.failure[child] = 0

                self.outputs[child].extend(self.outputs[self.failure[child]])

    def search(self, text: str) -> dict[str, list[int]]:
        results: dict[str, list[int]] = {pattern: [] for pattern in sum(self.outputs, [])}
        current = 0

        for i, char in enumerate(text):
            while current != 0 and char not in self.nodes[current]:
                current = self.failure[current]

            if char in self.nodes[current]:
                current = self.nodes[current][char]

            for pattern in self.outputs[current]:
                results[pattern].append(i - len(pattern) + 1)

        return results
